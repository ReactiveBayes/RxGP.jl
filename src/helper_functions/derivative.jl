# Define gradient functions for hyperparameter optimization

export neg_log_backwardmess_fast, neg_log_backwardmess_uncertain, grad_llh_default!, grad_llh_new!, grad_llh_uncertain!
export neg_log_backwardmess_multi, grad_llh_multi, grad_llh_multi!

## The following functions are copied from the backward message toward "θ" edge
# 1) univariate case
#for regression/classification with known input with arbitrary mean function
function neg_log_backwardmess_fast(θ; ω_data::Union{Nothing,AbstractVector{<:AbstractVector}}=nothing, y_data::Union{Nothing,AbstractVector}=nothing, x_data::AbstractVector{<:AbstractVector}, qv, qw=nothing, qWg=nothing, kernel, Ex, Dxθ, Cxθ_Xu, mean_fn, Xu)
    Kuu = kernelmatrix(kernel(θ), Xu) + 1e-8 * I
    mf = mean_fn
    mxu = apply_mean_fn.(Xu, mf)
    Ku_mxu = Kuu \ mxu
    mxuT_KuT = transpose(Ku_mxu)
    μ_v, Σ_v = mean_cov(qv)
    Rv = Σ_v + μ_v * μ_v'
    llh = 0.0

    ## ========== y data ============ ##
    if !isnothing(y_data)
        Kuu_inv = cholinv(Kuu)
        kxx = kernelmatrix_diag(kernel(θ), x_data)
        Kxu = kernelmatrix(kernel(θ), x_data, Xu)
        w_bar = mean(qw)
        mx_ = apply_mean_fn.(x_data, mf)
        μ_y_ = typeof(y_data[1]) == Union{MultivariateNormalDistributionsFamily, PointMass} ? mean.(y_data) : y_data
        for i in eachindex(y_data)
            mx = mx_[i]
            μ_y = μ_y_[i]

            Ψ0 = kxx[i]
            Ψ1 = Kxu[i:i, :] # Ψ1 = E_x[Bx] ≈ Kxu[i:i, :]
            Ψ2 = transpose(Kxu[i:i, :]) * Kxu[i:i, :]

            I1 = Ψ0 - tr( Kuu_inv * Ψ2 ) # I1 = E_x[Ax] ≈ kxx[i] - Kxu[i:i, :] * Kuu_inv * transpose(Kxu[i:i, :])
            part_A = tr(Rv * Ψ2) + dot(mxuT_KuT, Ψ2*Ku_mxu) + 2*mx*dot(Ψ1, μ_v) - 2*mx*dot(Ψ1, Ku_mxu) - 2*dot(transpose(μ_v), Ψ2*Ku_mxu)
            I5 = -2*μ_y*dot(Ψ1, (μ_v - Ku_mxu)) + part_A

            llh += - 0.5 * w_bar * ( I1 + I5 )
        end
    end

    ## ========== ω data ============ ##
    if !isnothing(ω_data)
        μ_ω_ = typeof(ω_data[1]) == Union{MultivariateNormalDistributionsFamily, PointMass} ? mean.(ω_data) : ω_data
        μ_x_ = typeof(x_data[1]) == Union{MultivariateNormalDistributionsFamily, PointMass} ? mean.(x_data) : x_data

        Wg_bar = mean(qWg)
        # Ex = (x) -> Ex(x)
        Dx = (x) -> Dxθ(x, θ)
        Cx = (x) -> Cxθ_Xu(x, θ, Xu)
        Ωx_ = Ex.(μ_x_)
        Ω0_ = Dx.(μ_x_)
        Ω1_ = Cx.(μ_x_)
        for i in eachindex(ω_data)
            Ωx = Ωx_[i]
            Ω0 = Ω0_[i]
            Ω1 = Ω1_[i]
            Ω3 = transpose(Ω1) * Wg_bar * Ω1
            Ω4 = transpose(Ωx) * Wg_bar * Ω1

            G1 = Ω0 - Ω1 * Kuu_inv * transpose(Ω1)
            part_A = 2 * dot(Ω4, (μ_v - Ku_mxu)) + dot((mxuT_KuT - 2*transpose(μ_v)), Ω3*Ku_mxu) + tr(Ω3 * Rv)
            part_B = 2 * dot(transpose(μ_ω_[i]), Wg_bar * Ω1 * (μ_v - Ku_mxu))

            llh += - 0.5 * tr( Wg_bar * G1 ) - 0.5 * ( part_A - part_B )
        end
    end

    return -llh
end

#case that input is random variable 
function neg_log_backwardmess_uncertain(θ; y_data, qx,v, Uv,w,kernel,Xu,method)
    Kuu_inverse = inv(kernelmatrix(kernel(θ), Xu) + 1e-12*I) 
    llh = 0.0
    @inbounds for i in eachindex(y_data)
        Ψ0 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], [x]),qx[i])[1]
        Ψ1_trans = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu,[x]),qx[i]) 
        Ψ2 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), qx[i])
        llh += w * y_data[i] * dot(Ψ1_trans, v) - 0.5 * w * (Ψ0 + tr(Ψ2*(Uv'*Uv - Kuu_inverse)))
    end
    return -llh
end

## Define the corresponding derivative (gradient) functions 
function grad_llh_default!(grad, θ; ω_data=nothing, y_data=nothing, x_data, qv, qw=nothing, qWg=nothing, kernel, Ex=nothing, Dxθ=nothing, Cxθ_Xu=nothing, mean_fn=nothing, Xu)
    return ForwardDiff.gradient!(grad, (x) -> neg_log_backwardmess_fast(x; ω_data=ω_data, y_data=y_data, x_data=x_data, qv=qv, qw=qw, qWg=qWg, kernel=kernel, Ex=Ex, Dxθ=Dxθ, Cxθ_Xu=Cxθ_Xu, mean_fn=mean_fn, Xu=Xu), θ)
end

#this is for big data
function grad_llh_new!(grad, θ; y_data, x_data, qv, qw, kernel, Xu, chunk_size)
    newfunc = (x) -> neg_log_backwardmess_fast(x;y_data=y_data,x_data=x_data,qv=qv,qw=qw,kernel=kernel,Xu=Xu)
    cfg = GradientConfig(newfunc, θ, Chunk{chunk_size}())
    return ForwardDiff.gradient!(grad, newfunc, θ,cfg)
end

function grad_llh_uncertain!(grad,θ; y_data, qx,v, Uv,w,kernel,Xu,method)
    return ForwardDiff.gradient!(grad, (x) -> neg_log_backwardmess_uncertain(x; y_data=y_data, qx=qx,v=v, Uv=Uv,w=w,kernel=kernel,Xu=Xu,method=method), θ)
end

# 2) multivariate case
"""
this function is only for C = I
    y_data: Array of mean.(qy), or array of output 
    qx : Array of distributions 
    sumRv_Wbar : Real, sum(Rv_blk .* W)
    v : mean(qv)
    W : mean(qW)
    Tr_W : trace(W)
    kernel : kernel of gp 
    Xu : inducing points 
    method: method to approximate kernel expectation 
"""
function neg_log_backwardmess_multi(θ;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method)
        Kuu_inverse = inv(kernelmatrix(kernel(θ), Xu) + 1e-12*I)        
        llh = 0.0
        M = size(Xu,1)
        @inbounds for i in eachindex(qx)
            @inbounds V = v * y_data[i]' * W
            sumdiagV = sum_diagonal_M(V,M)
            @inbounds Ψ_0 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], [x]),qx[i])[1] # kxx
            @inbounds Ψ_1_trans = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu,[x]),qx[i]) # kux
            @inbounds Ψ_2 = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), qx[i]) # kux*kxu 

            llh += -0.5 * tr_W * (Ψ_0 - sum(Kuu_inverse .* Ψ_2)) + sum(sumdiagV .* Ψ_1_trans) - 0.5 * sum(sumRv_Wbar .* Ψ_2)
        end
    return -llh
end

function grad_llh_multi(θ;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method)
    return ForwardDiff.gradient((x) -> neg_log_backwardmess_multi(x;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method), θ)
end

function grad_llh_multi!(grad,θ;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method)
    return ForwardDiff.gradient!(grad,(x) -> neg_log_backwardmess_multi(x;y_data, qx, sumRv_Wbar, v, W ,tr_W, kernel, Xu,method), θ)
end

function sum_diagonal_M(V, M)
    @inbounds sumΣV = sum(view(V,M*(i-1)+1:i*M,i)  for i=1:size(V,2))
    return sumΣV
end

#compute trace of block matrix 
function trace_blkmatrix(Rv,D,M)
    return [tr(view(Rv,i:i+M-1,j:j+M-1)) for i=1:M:M*D, j=1:M:M*D]
end