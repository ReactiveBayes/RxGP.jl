export jdotavx, create_blockmatrix
export get_gradient_Kxu_fn, get_gradient_Kxx_fn, get_UniSGPMeta, get_simple_kernel_and_params, apply_mean_fn, mean_cov_scalar_matrix, mean_cov_vector_matrix

# Keep kernel params strictly positive while preserving AD stability.
softplus_pos(x) = StatsFuns.softplus(x) + eps(Float64)


# ======= Build operator-specific kernel and mean functions ======= #
# Private helper: gradient of m(x) w.r.t. x, returns a P-vector.
function _build_mean_gradient_fn(mean_fn)
    return (x) -> begin
        if x isa Distribution
            x = mean(x)
        end
        if x isa Number
            return [ForwardDiff.derivative(z -> mean_fn(z), x)]
        elseif x isa AbstractVector
            if length(x) == 1
                return [ForwardDiff.derivative(z -> mean_fn(z), x[1])]
            else
                return ForwardDiff.gradient(z -> mean_fn(z), x)
            end
        else
            error("Type of x: $(typeof(x)) not supported in _build_mean_gradient_fn")
        end
    end
end

# Build the three operator functions (Lm_fn, Kxu_fn, Kxx_fn) for a given linear operator.
# operator :fn              — identity operator (P=1)
# operator :grad            — gradient operator (P=D)
# operator :joint_fn_grad   — stacked identity+gradient (P=1+D)
function _build_operator_fns(D; mean_fn, kernel, kernel_spec, mode, operator::Symbol, independent_SE_lengthscales::Bool)
    if operator == :fn
        # Identity operator: ϕ̃(s) collapses to the standard scalar/vector VSGP observation model.
        Lm_fn  = (x) -> [apply_mean_fn(x, mean_fn)]
        Kxu_fn = (x, θ, Xu) -> kernelmatrix(kernel(θ), [x], Xu)    # 1×M
        Kxx_fn = (x, θ) -> kernelmatrix(kernel(θ), [x], [x])        # 1×1
        return Lm_fn, Kxu_fn, Kxx_fn, 1

    elseif operator == :grad
        # Gradient operator ℒ = ∇_x gives P=D dimensional observations.
        @assert (mode == :AD && kernel_spec in (:SE, :SEn, :SMn, :SEn_SMn)) || (mode == :AN && kernel_spec in (:SE, :SEn)) "For kernel_spec :SMn/:SEn_SMn mode must be :AD; for :SE/:SEn mode can be :AD or :AN"
        Lm_fn  = _build_mean_gradient_fn(mean_fn)
        Kxu_fn = get_gradient_Kxu_fn(D; kernel=kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
        Kxx_fn = get_gradient_Kxx_fn(D; kernel=kernel, kernel_spec=kernel_spec, mode=mode, independent_SE_lengthscales=independent_SE_lengthscales)
        return Lm_fn, Kxu_fn, Kxx_fn, D

    elseif operator == :joint_fn_grad
        # Stacked operator ℒ = [I; ∇_x]^T gives P = 1+D dimensional observations.
        # Kxu_fn and Kxx_fn for the joint case always use AD (no analytic shortcut for the cross-block).
        @assert (mode == :AD && kernel_spec in (:SE, :SEn, :SMn, :SEn_SMn)) || (mode == :AN && kernel_spec in (:SE, :SEn)) "For kernel_spec :SMn/:SEn_SMn mode must be :AD; for :SE/:SEn mode can be :AD or :AN"
        grad_m   = _build_mean_gradient_fn(mean_fn)
        grad_Kxu = get_gradient_Kxu_fn(D; kernel=kernel, kernel_spec=kernel_spec, mode=:AD, independent_SE_lengthscales=independent_SE_lengthscales)
        grad_Kxx = get_gradient_Kxx_fn(D; kernel=kernel, kernel_spec=kernel_spec, mode=:AD, independent_SE_lengthscales=independent_SE_lengthscales)

        Lm_fn = (x) -> vcat([apply_mean_fn(x, mean_fn)], grad_m(x))  # (1+D)-vector

        Kxu_fn = (x, θ, Xu) -> begin
            scalar_row = kernelmatrix(kernel(θ), [x], Xu)  # 1×M
            grad_rows  = grad_Kxu(x, θ, Xu)               # D×M
            vcat(scalar_row, grad_rows)                     # (1+D)×M
        end

        # ϕ̃ kernel covariance: full (1+D)×(1+D) block matrix = ℒ_1 ℒ_2 k(x,x)
        Kxx_fn = (x, θ) -> begin
            f = (x1, x2) -> kernelmatrix(kernel(θ), [x1], [x2])[1]
            kxx_val  = f(x, x)                                                  # scalar
            kxx_fwd  = kernel_spec in (:SE, :SEn, :SMn, :SEn_SMn) ? zeros(1, D) : reshape(ForwardDiff.gradient(z -> f(x, z), x), 1, D)  # 1×D  (∇_{x'} k(x,x)=0 for stationary kernels)
            kxx_bwd  = kernel_spec in (:SE, :SEn, :SMn, :SEn_SMn) ? zeros(D, 1) : reshape(ForwardDiff.gradient(z -> f(z, x), x), D, 1)  # D×1  (∇_x k(x,x)=0 for stationary kernels)
            kxx_hess = grad_Kxx(x, θ)                                           # D×D  (∇_x∇_{x'} k)
            vcat(hcat(kxx_val, kxx_fwd), hcat(kxx_bwd, kxx_hess))               # (1+D)×(1+D)
        end

        return Lm_fn, Kxu_fn, Kxx_fn, 1 + D
    else
        error("operator must be :fn, :grad, or :joint_fn_grad. Got: $operator")
    end
end

# ======= Single source of meta information ======= #
# operator — :fn (default), :grad, or :joint_fn_grad (see _build_operator_fns above).
# Custom Lm_fn / Kxu_fn / Kxx_fn can be provided to use an arbitrary linear operator;
# if any are given manually all three must be provided.
function get_UniSGPMeta(D; method=nothing, mean_fn, kernel, kernel_spec::Symbol=:SE, mode::Symbol=:AD,
                         operator::Symbol=:fn,
                         independent_SE_lengthscales::Bool=true, Xu, θ,
                         Lm_fn=nothing, Kxu_fn=nothing, Kxx_fn=nothing)
    θ = typeof(θ) <: PointMass ? mean(θ) : θ
    dims_theta = length(θ)
    Kuu = kernelmatrix(kernel(θ), Xu) + 1e-8 * I
    KuuF = cholesky(Kuu)
    x_dummy = zeros(D)
    Ψx = 0.0
    Ψxx = 0.0
    Ψ0 = kernelmatrix(kernel(θ), [x_dummy])[1]
    Ψ1_trans = kernelmatrix(kernel(θ), Xu, [x_dummy])
    Ψ2 = kernelmatrix(kernel(θ), Xu, [x_dummy]) * kernelmatrix(kernel(θ), [x_dummy], Xu)
    Ψ3 = kernelmatrix(kernel(θ), [x_dummy], Xu)
    Uv = zeros(size(Xu, 1), size(Xu, 1))

    if Lm_fn === nothing || Kxu_fn === nothing || Kxx_fn === nothing
        (Lm_fn_default, Kxu_fn_default, Kxx_fn_default, dims_data) =
            _build_operator_fns(D; mean_fn=mean_fn, kernel=kernel, kernel_spec=kernel_spec,
                                mode=mode, operator=operator, independent_SE_lengthscales=independent_SE_lengthscales)
        Lm_fn  = Lm_fn  === nothing ? Lm_fn_default  : Lm_fn
        Kxu_fn = Kxu_fn === nothing ? Kxu_fn_default : Kxu_fn
        Kxx_fn = Kxx_fn === nothing ? Kxx_fn_default : Kxx_fn
    else
        # All three provided manually — infer dims_data from a test call
        dims_data = length(Lm_fn(x_dummy))
    end

    return UniSGPMeta(method, mean_fn, Xu, Ψx, Ψxx, Ψ0, Ψ1_trans, Ψ2, Ψ3, Lm_fn, Kxx_fn, Kxu_fn, KuuF, kernel, dims_data, dims_theta, Uv, 0, 1)
end


# # ======= Kernel and Mean-Function Diff Functions - GENERAL ======= #
# Gradient of k(x, Xu_j) w.r.t x — produces K̃_xu(x,θ) ∈ ℝ^{D×M} for the gradient operator.
function get_gradient_Kxu_fn(D; kernel::Any=kernel, kernel_spec::Symbol=:SE, mode::Symbol=:AD, independent_SE_lengthscales::Bool=true)
    if mode == :AD
        # AutoDiff approach
        return (x, θ, Xu) -> begin
            D = length(x)
            N = length(Xu)
            f = (x1, x2) -> kernelmatrix(kernel(θ), [x1], [x2])[1]
            res = [ForwardDiff.gradient(z -> f(Xu[j], z), x)[i] for i in 1:D, j in 1:N]
            if any(isnan, res)
                @warn "NaN encountered in gradient_Kxu_fn, returning gradient_Kxu_fn with spread points" maxlog=1
                return [ForwardDiff.gradient(z -> f(Xu[j] .- 5e-7, z), x .+ 5e-7)[i] for i in 1:D, j in 1:N]
            end
            return res
        end

    elseif mode == :AN
        # Analytic approach (for SE/SEn kernel only) ~ 10x faster than AutoDiff
        if kernel_spec == :SE && !independent_SE_lengthscales
            return (x, θ, Xu) -> begin
                Lambda_inv = 1/(softplus_pos.(θ[2])^2) * Matrix(I,D,D)
                tX = hcat(Xu...)' .- x'
                Kxu = kernelmatrix(kernel(θ), [x], Xu)
                return (transpose(Lambda_inv) * transpose(tX)) .* Kxu
            end
        elseif kernel_spec == :SE && independent_SE_lengthscales
            return (x, θ, Xu) -> begin
                Lambda_inv = inv((softplus_pos.(θ[2:end]) .* Matrix(I,D,D))^2)
                tX = hcat(Xu...)' .- x'
                Kxu = kernelmatrix(kernel(θ), [x], Xu)
                return (transpose(Lambda_inv) * transpose(tX)) .* Kxu
            end
        elseif kernel_spec == :SEn && !independent_SE_lengthscales
            return (x, θ, Xu) -> begin
                dim_θ = length(θ)
                N = length(Xu)
                @assert iseven(dim_θ)
                w_ = θ[1:div(dim_θ,2)]
                l_ = θ[div(dim_θ,2)+1:end]
                tXT = transpose(hcat(Xu...)' .- x')
                k = (w, l) -> softplus_pos(w) * with_lengthscale(SEKernel(), softplus_pos(l))
                Kxu = (w, l) -> kernelmatrix(k(w,l), [x], Xu)
                Lambda_inv_T = (l) -> transpose(inv((softplus_pos.(l) .* Matrix(I,D,D))^2))
                return sum((Lambda_inv_T(l) * tXT) .* Kxu(w,l) for (w,l) in zip(w_,l_))
            end
        elseif kernel_spec == :SEn && independent_SE_lengthscales
            return (x, θ, Xu) -> begin
                @assert length(θ) % (D+1) == 0
                num_SE = div(length(θ), D+1)
                N = length(Xu)
                tXT = transpose(hcat(Xu...)' .- x')
                T = promote_type(eltype(x), eltype(θ))
                res = zeros(T, D, N)
                for i in 1:num_SE
                    wi = softplus_pos(θ[(i-1)*(D+1)+1])
                    li = softplus_pos.(θ[(i-1)*(D+1)+2 : i*(D+1)])
                    Λinv = Diagonal(1 ./ li.^2)
                    Ki = kernelmatrix(
                        wi * with_lengthscale(SEKernel(), li),
                        [x], Xu
                    )
                    res .+= (Λinv' * tXT) .* Ki
                end
                return res
            end
        else
            error("Currently only SE and SE2 kernels supported for analytic gradient_Kxu_fn")
        end
    else
        error("mode must be :AD (AutoDiff) or :AN (Analytic)")
    end
end

# Hessian of k(x, x) w.r.t x — produces K̃_xx(x,θ) ∈ ℝ^{D×D} for the gradient operator.
function get_gradient_Kxx_fn(D; kernel::Any=kernel, kernel_spec::Symbol=:SE, mode::Symbol=:AD, independent_SE_lengthscales::Bool=true)
    if mode == :AD
        # AutoDiff approach
        return (x, θ) -> begin
            f = (x1, x2) -> kernelmatrix(kernel(θ), [x1], [x2])[1]
            g = (x1, x2) -> ForwardDiff.gradient(z -> f(z, x2), x1)
            res = ForwardDiff.jacobian(z -> g(x, z), x)
            if any(isnan, res)
                @warn "NaN encountered in gradient_Kxx_fn, returning gradient_Kxx_fn with spread points" maxlog=1
                return ForwardDiff.jacobian(z -> g(x .- 5e-7, z), x .+ 5e-7)
            end
            return res
        end

    elseif mode == :AN
        # Analytic approach (for SE/SEn kernel only) ~ 10x faster than AutoDiff
        if kernel_spec == :SE && !independent_SE_lengthscales
            return (x, θ) -> begin
                Lambda_inv = 1/(softplus_pos.(θ[2])^2) * Matrix(I,D,D)
                return Lambda_inv * softplus_pos(θ[1])
            end
        elseif kernel_spec == :SE && independent_SE_lengthscales
            return (x, θ) -> begin
                Lambda_inv = inv((softplus_pos.(θ[2:end]) .* Matrix(I,D,D))^2)
                return Lambda_inv * softplus_pos(θ[1])
            end
        elseif kernel_spec == :SEn && !independent_SE_lengthscales
            return (x, θ) -> begin
                dim_θ = length(θ)
                @assert iseven(dim_θ)
                w = θ[1:div(dim_θ,2)]
                l = θ[div(dim_θ,2)+1:end]
                T = promote_type(eltype(x), eltype(θ))
                sum = zeros(T, D, D)
                for (wi, li) in zip(w, l)
                    Lambda_invi = inv((softplus_pos.(li) .* Matrix(I,D,D))^2)
                    sum += Lambda_invi * softplus_pos(wi)
                end
                return sum
            end
        elseif kernel_spec == :SEn && independent_SE_lengthscales
            return (x, θ) -> begin
                @assert length(θ) % (D+1) == 0
                num_SE = div(length(θ), D+1)
                T = promote_type(eltype(x), eltype(θ))
                sum = zeros(T, D, D)
                for i in 1:num_SE
                    wi = softplus_pos(θ[(i-1)*(D+1)+1])
                    li = softplus_pos.(θ[(i-1)*(D+1)+2 : i*(D+1)])
                    Λinv = Diagonal(1 ./ li.^2)
                    sum .+= wi * Λinv
                end
                return sum
            end
        else
            error("Currently only SE and SEn kernels supported for analytic gradient_Kxx_fn")
        end
    else
        error("mode must be :AD (AutoDiff) or :AN (Analytic)")
    end
end

# # ================== Simple KernelFunctions.jl Kernel Builder (dimension-agnostic) ================== #
function get_simple_kernel_and_params(D; kernel_spec::Symbol=:SE, num_SE::Int=1, num_SM::Int=1, independent_SE_lengthscales::Bool=true)
    """
        get_simple_kernel_and_params(D; kernel_spec=:SE, num_SE=1, num_SM=1)

    This function returns a simple KernelFunctions.jl kernel and its parameters for a given dimensionality and kernel specification.

    # Arguments
    - `D::Int`: Dimensionality of the input data.
    - `kernel_spec::Symbol`: Type of kernel to create. Options are:
        - `:SE`: Squared Exponential Kernel
        - `:SEn`: Mixture of num_SE Squared Exponential Kernels
        - `:SMn`: Spectral Mixture Kernel with num_SM components
        - `:SEn_SMn`: Mixture of num_SE Squared Exponential Kernels + Spectral Mixture Kernel with num_SM components
    - `num_SE::Int`: Number of components for the SE mixture (used if `kernel_spec` is `:SEn` or `:SEn_SMn`).
    - `num_SM::Int`: Number of components for the Spectral Mixture kernel (used if `kernel_spec` is `:SMn` or `:SEn_SMn`).
    # Returns
    - `kernel`: A callable KernelFunctions.jl kernel.
    - `θ_init`: Initial parameters for the kernel.
    - `dim_θ`: Dimensionality of the kernel parameters.
    """
    if kernel_spec == :SE
        # SEKernel
        SE = independent_SE_lengthscales ? D + 1 : 2 # number of params (1 weight + D lengthscales or 1 shared lengthscale)
        dim_θ = SE; # number of kernel parameters
        θ_init = StatsFuns.invsoftplus.(ones(dim_θ) .+ collect(range(-0.05, 0.05, length=dim_θ)));
        kernel = (θ) -> softplus_pos(θ[1]) * with_lengthscale(SEKernel(), softplus_pos.(θ[2:SE]))
        return kernel, θ_init, dim_θ

    elseif kernel_spec == :SEn && !independent_SE_lengthscales
        # SEKernel Mixture with shared lengthscales
        SE = 2 # number of params per component (1 weight + 1 lengthscale)
        dim_θ = SE * num_SE; # number of kernel parameters
        θ_init = StatsFuns.invsoftplus.(ones(dim_θ) .+ collect(range(-0.05, 0.05, length=dim_θ)));
        @assert iseven(dim_θ) "dim_θ must be even for SEKernel mixture"
        half_dim_θ = div(dim_θ, 2)
        kernel = (θ) -> begin 
            w = θ[1:half_dim_θ];
            l = θ[half_dim_θ+1:end];
            kernel_i = (wi, li) -> softplus_pos(wi) * with_lengthscale(SEKernel(), softplus_pos(li))
            kernels = []
            for (wi, li) in zip(w,l)
                push!(kernels, kernel_i(wi, li))
            end
            return KernelSum(kernels)
        end
        return kernel, θ_init, dim_θ

    elseif kernel_spec == :SEn && independent_SE_lengthscales
        # SEKernel Mixture with independent lengthscales
        dim_θ = num_SE * (D+1)
        θ_init = StatsFuns.invsoftplus.(ones(dim_θ) .+ collect(range(-0.05, 0.05, length=dim_θ)))
        kernel = (θ) -> begin
            kernels = [
                softplus_pos(θ[(i-1)*(D+1)+1]) *
                with_lengthscale(SEKernel(), softplus_pos.(θ[(i-1)*(D+1)+2 : i*(D+1)]))
                for i in 1:num_SE
            ]
            return KernelSum(kernels)
        end
        return kernel, θ_init, dim_θ

    elseif kernel_spec == :SMn
        # Spectral Mixture Kernel
        SM = num_SM * (2 * D + 1)  # params per SM component
        dim_θ = SM # number of kernel parameters
        θ_init = StatsFuns.invsoftplus.(ones(dim_θ) .+ collect(range(-0.05, 0.05, length=dim_θ)))
        # Spectral mixture with num_SM components
        kernel_SM = (θ) -> begin
            weights = softplus_pos.(θ[1:num_SM])
            covariance = reshape(softplus_pos.(θ[num_SM+1 : num_SM + D*num_SM]), (D, num_SM))
            means = reshape(softplus_pos.(θ[num_SM + D*num_SM + 1:end]), (D, num_SM))
            KernelFunctions.spectral_mixture_kernel(weights, covariance, means)
        end
        kernel = (θ) -> kernel_SM(θ)
        return kernel, θ_init, dim_θ

    elseif kernel_spec == :SEn_SMn
        # SEKernel Mixture + Spectral Mixture Kernel
        SE = independent_SE_lengthscales ? num_SE * (D + 1) : num_SE * 2  # params per SE component (weight + lengthscale(s))
        SM = num_SM * (2 * D + 1)  # params per SM component
        dim_θ = SE + SM # number of kernel parameters
        θ_init = StatsFuns.invsoftplus.(ones(dim_θ) .+ collect(range(-0.05, 0.05, length=dim_θ)))
        
        # Mixture of num_SE SE kernels
        kernel_SE = (θ) -> begin
            if !independent_SE_lengthscales
                kernels = [
                    softplus_pos(θ[2i-1]) * with_lengthscale(SEKernel(), softplus_pos(θ[2i]))
                    for i in 1:num_SE
                ]
                return KernelSum(kernels)
            else
                kernels = [
                    softplus_pos(θ[(i-1)*(D+1)+1]) *
                    with_lengthscale(SEKernel(), softplus_pos.(θ[(i-1)*(D+1)+2 : i*(D+1)]))
                    for i in 1:num_SE
                ]
                return KernelSum(kernels)
            end
        end

        # Spectral mixture with num_SM components
        kernel_SM = (θ) -> begin
            weights = softplus_pos.(θ[1:num_SM])
            covariance = reshape(softplus_pos.(θ[num_SM+1 : num_SM + D*num_SM]), (D, num_SM))
            means = reshape(softplus_pos.(θ[num_SM + D*num_SM + 1:end]), (D, num_SM))
            KernelFunctions.spectral_mixture_kernel(weights, covariance, means)
        end

        # Combined kernel
        kernel = (θ) -> kernel_SE(θ[1:SE]) + kernel_SM(θ[SE+1:end])
        return kernel, θ_init, dim_θ
    end
end


# ==== ======= Mean Function Application Helper (handles scalars, vectors, vectors of vectors) ======= #
"""
    apply_mean_fn(x, mf)
Apply a SCALAR mean function `mf` to input `x`, handling different input types:
- If `x` is a scalar, apply `mf` directly. E.g. for mf(x) = x^2, apply_mean_fn(3, mf) returns 9.
- If `x` is a length-1 vector of numbers, apply `mf` to the single element.
- If `x` is a longer vector of numbers, apply `mf` to the entire vector.
- If `x` is a vector of vectors, use apply_mean_fn.(x, mf) to apply `mf` to each sub-vector.
"""
apply_mean_fn(x::Number, mf) = mf(x) # scalar
apply_mean_fn(x::AbstractVector{<:Number}, mf) = begin length(x) == 1 ? mf(x[1]) : mf(x) end # length-1 vector of numbers


# ==================== Expanded mean_cov functionality to enforce return types ======================== #
"""
    mean_cov_scalar_matrix(x::Union{Real, AbstractVector, PointMass, UnivariateNormalDistributionsFamily})

    Dimensionality of x must be 1. Return mean as scalar and covariance as matrix.

    args:
        x::Union{Real, AbstractVector, PointMass, UnivariateNormalDistributionsFamily}

    returns:
        (mean::Number, cov::Matrix)
"""
mean_cov_scalar_matrix(x::Real) = (x, [0.0;;]) # Base case: scalar number
mean_cov_scalar_matrix(x::UnivariateNormalDistributionsFamily) = (mean(x), [var(x);;]) # Univariate normal distributions
mean_cov_scalar_matrix(x::PointMass{<:Real}) = (mean(x), [0.0;;]) # PointMass of a scalar
# 1-element vector containing a scalar
function mean_cov_scalar_matrix(x::AbstractVector{<:Real})
    @assert length(x) == 1 "x must be scalar (e.g. length 1 vector)"
    (x[1], [0.0;;])
end
# 1-element vector containing a distribution/pointmass
function mean_cov_scalar_matrix(x::AbstractVector{<:Union{PointMass,UnivariateNormalDistributionsFamily}})
    @assert length(x) == 1 "x must be scalar (e.g. length 1 vector)"
    mean_cov_scalar_matrix(x[1])
end
# PointMass of a 1-element vector
function mean_cov_scalar_matrix(x::PointMass{<:AbstractVector{<:Real}})
    μ = mean(x)
    @assert length(μ) == 1 "x must be scalar (e.g. length 1 PointMass)"
    (μ[1], [0.0;;])
end

"""
    mean_cov_vector_matrix(x::Union{Real, AbstractVector, PointMass, NormalDistributionsFamily})

    Regardless of dimensionality of x, return mean as vector and covariance as matrix.

    args:
        x::Union{Real, AbstractVector, PointMass, NormalDistributionsFamily}

    returns:
        (mean::AbstractVector, cov::Matrix)
"""
mean_cov_vector_matrix(x::Real) = ([x], [0.0;;]) # Real scalar
mean_cov_vector_matrix(x::UnivariateNormalDistributionsFamily) = ([mean(x)], [var(x);;]) # Univariate normal distribution
mean_cov_vector_matrix(x::MultivariateNormalDistributionsFamily) = mean_cov(x) # Multivariate normal distribution
mean_cov_vector_matrix(x::AbstractVector{<:Real}) = (x, zeros(eltype(x), length(x), length(x))) # Plain vector of reals
mean_cov_vector_matrix(x::PointMass{<:Real}) = ([mean(x)], [0.0;;]) # PointMass around a real scalar
mean_cov_vector_matrix(x::PointMass{<:AbstractVector{<:Real}}) = (mean(x), zeros(eltype(mean(x)), length(mean(x)), length(mean(x)))) # PointMass around a vector of reals
# 1-element vector containing a distribution/pointmass
function mean_cov_vector_matrix(x::AbstractVector{<:Union{PointMass,UnivariateNormalDistributionsFamily,MultivariateNormalDistributionsFamily}})
    if length(x) > 1
        error("Unsupported input: Input should represent a single item, not a list of items")
    end
    mean_cov_vector_matrix(x[1])
end


# ==================== Miscellaneous ======================== #
function jdotavx(a::AbstractArray, b::AbstractArray)
    a_flat, b_flat = collect(vec(a)), collect(vec(b))
    @assert axes(a_flat) == axes(b_flat)
    @assert length(a_flat) == length(b_flat) "dot operands must have same number of elements"
    s = zero(promote_type(eltype(a), eltype(b)))
    @turbo warn_check_args=false for i in eachindex(a_flat, b_flat)
        s += a_flat[i] * b_flat[i]
    end
    return s
end

function create_blockmatrix(A,d,M)
    return [view(A,i:i+M-1,j:j+M-1) for i=1:M:M*d, j=1:M:M*d]
end