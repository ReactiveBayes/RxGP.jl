export predict_GP_values, predict_GP_gradients, predict_GP_joints

#---- Define prediction functions ----#

"""
    predict_GP_values(m_in, q_v, q_θ, meta)

Predict the univariate GP latent function values at new inputs, returning the predictive mean and marginal variance per input using the sparse GP posterior over inducing variables.

Arguments
    m_in: vector of real-valued vector data-points, so, e.g. [[1,2,3], [4,5,6], ...] or [[1], [2], [3], ...]
    q_v::MultivariateNormalDistributionsFamily: posterior over inducing variables; its mean and covariance are used.
    q_θ::Any: kernel hyperparameters; if a distribution (PointMass or MVN), its mean is used.
    meta::UniSGPMeta: model metadata (mean function, inducing inputs Xu, kernel, precomputed Kuu factors).

Returns
    (means, vars)
        means::Vector{<:Real}: predictive means m_f for each input in m_in.
        vars::Vector{<:Real}: predictive marginal variances V_f for each input in m_in.
"""
function predict_GP_values(; m_in::AbstractVector{<:AbstractVector{<:Real}}, q_v::MultivariateNormalDistributionsFamily, q_θ::Any, meta::UniSGPMeta,)
    θ = typeof(q_θ) <: Union{MultivariateNormalDistributionsFamily, PointMass} ? mean(q_θ) : q_θ
    μ_v, Σ_v = mean_cov(q_v)
    mf = getMeanFn(meta)
    Xu = getInducingInput(meta)
    
    kernel = getKernel(meta)

    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    
    mx_ = apply_mean_fn.(m_in, mf)
    kxx = kernelmatrix_diag(kernel(θ), m_in)
    Kxu = kernelmatrix(kernel(θ), m_in, Xu)

    predictions_mean_ = Float64[]
    predictions_var_ = Float64[]

    for i in eachindex(m_in)
        Ψ0 = [kxx[i];;]
        Ψ1 = Kxu[i:i, :]
        Ψ1_trans = transpose(Ψ1)

        m_f = mx_[i] + dot(Ψ1, (μ_v - Ku_mxu))
        Kuu_inv_Ψ1 = meta.KuuF \ Ψ1_trans
        V_f = (Ψ0 + Ψ1 * Σ_v * Ψ1_trans - Ψ1 * Kuu_inv_Ψ1)[1]

        push!(predictions_mean_, m_f)
        push!(predictions_var_, V_f)
    end

    return predictions_mean_, predictions_var_
end

"""
    predict_GP_gradients(m_in, q_v, q_θ, meta)

Predict the gradient of the univariate GP latent function at new inputs, returning the predictive gradient mean and covariance per input.

Arguments
    m_in: vector of real-valued vector data-points, so, e.g. [[1,2,3], [4,5,6], ...] or [[1], [2], [3], ...]
    q_v::MultivariateNormalDistributionsFamily: posterior over inducing variables; its mean and covariance are used.
    q_θ::Any: kernel hyperparameters; if a distribution (PointMass or MVN), its mean is used.
    meta::UniSGPMeta: metadata providing Ex, Dxθ, Cxθ_Xu, inducing inputs Xu, and Kuu factors.

Returns
    (grad_means, grad_covs)
        grad_means::Vector{<:AbstractVector}: gradient mean vector m_g for each input (dimension equals input dimension).
        grad_covs::Vector{<:AbstractMatrix}: gradient covariance matrix C_g for each input (d×d).
"""
function predict_GP_gradients(; m_in::AbstractVector{<:AbstractVector{<:Real}}, q_v::MultivariateNormalDistributionsFamily, q_θ::Any, meta::UniSGPMeta,)
    θ = typeof(q_θ) <: Union{MultivariateNormalDistributionsFamily, PointMass} ? mean(q_θ) : q_θ
    μ_v, Σ_v = mean_cov(q_v)
    mf = getMeanFn(meta)
    Xu = getInducingInput(meta)
    
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)
    Dx = (x) -> Dxθ(x, θ)
    Cx = (x) -> Cxθ_Xu(x, θ, Xu)

    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    Kuu_inv = meta.KuuF \ I

    E_ = Ex.(m_in)
    D_ = Dx.(m_in)
    C_ = Cx.(m_in)

    predictions_mean_ = []
    predictions_cov_ = Matrix{Float64}[]

    for i in eachindex(m_in)
        E = E_[i]
        D = D_[i]
        C = C_[i]

        m_g = E + C * (μ_v - Ku_mxu)
        Kuu_inv_C = meta.KuuF \ transpose(C)
        C_g = D + C * Σ_v * transpose(C) - C * Kuu_inv_C

        push!(predictions_mean_, m_g)
        push!(predictions_cov_, C_g)
    end

    return predictions_mean_, predictions_cov_
end

"""
    predict_GP_joints(m_in, q_v, q_θ, meta)

Predict the joints of the univariate GP latent function value and its gradient at new inputs, returning a stacked mean [f; g] and the full block covariance per input.

Arguments
    m_in: vector of real-valued vector data-points, so, e.g. [[1,2,3], [4,5,6], ...] or [[1], [2], [3], ...]
    q_v::MultivariateNormalDistributionsFamily: posterior over inducing variables; its mean and covariance are used.
    q_θ::Any: kernel hyperparameters; if a distribution (PointMass or MVN), its mean is used.
    meta::UniSGPMeta: metadata providing mean/gradient helper functions, inducing inputs Xu, kernel, and Kuu factors.

Returns
    (means, covs)
        means::Matrix: N×(1+d) matrix where each row is [m_f, m_g...], for N = length(m_in) and input dimension d.
        covs::Array{<:Real,3}: N×(1+d)×(1+d) stack of joint covariance matrices with blocks [V_f C_fg; C_gf C_g].
"""
function predict_GP_joints(; m_in::AbstractVector{<:AbstractVector{<:Real}}, q_v::MultivariateNormalDistributionsFamily, q_θ::Any, meta::UniSGPMeta,)
    θ = typeof(q_θ) <: Union{MultivariateNormalDistributionsFamily, PointMass} ? mean(q_θ) : q_θ
    μ_v, Σ_v = mean_cov(q_v)
    mf = getMeanFn(meta)
    Xu = getInducingInput(meta)
    
    Ex = getEx(meta)
    Fxθ = getFxθ(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)
    Fx = (x) -> Fxθ(x, θ)
    Dx = (x) -> Dxθ(x, θ)
    Cx = (x) -> Cxθ_Xu(x, θ, Xu)
    kernel = getKernel(meta)

    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    Kuu_inv = meta.KuuF \ I

    mx_ = apply_mean_fn.(m_in, mf)
    kxx = kernelmatrix_diag(kernel(θ), m_in)
    Kxu = kernelmatrix(kernel(θ), m_in, Xu)

    E_ = Ex.(m_in)
    F_ = Fx.(m_in)
    D_ = Dx.(m_in)
    C_ = Cx.(m_in)

    predictions_mean_ = Vector{Float64}[]
    predictions_cov_ = Matrix{Float64}[]

    for i in eachindex(m_in)
        Ψ0 = [kxx[i];;]
        Ψ1 = Kxu[i:i, :]
        Ψ1_trans = transpose(Ψ1)

        E = E_[i]
        F = F_[i]
        D = D_[i]
        C = C_[i]

        m_f = mx_[i] + dot(Ψ1, (μ_v - Ku_mxu))
        Kuu_inv_Ψ1 = meta.KuuF \ Ψ1_trans
        V_f = (Ψ0 + Ψ1 * Σ_v * Ψ1_trans - Ψ1 * Kuu_inv_Ψ1)[1]

        m_g = E + C * (μ_v - Ku_mxu)
        Kuu_inv_C = meta.KuuF \ transpose(C)
        C_g = D + C * Σ_v * transpose(C) - C * Kuu_inv_C
        C_fg = 2 * F + 2 * Ψ1 * Σ_v * transpose(C) - 2 * Ψ1 * Kuu_inv_C
        C_gf = transpose(C_fg)

        m_j = [m_f; m_g]
        C_j = [V_f C_fg; C_gf C_g]

        append!(predictions_mean_, [m_j])
        append!(predictions_cov_, [C_j])
    end
    
    predictions_mean = hcat(predictions_mean_...)'
    predictions_cov = permutedims(cat(predictions_cov_..., dims=3), (3,1,2))
    return predictions_mean, predictions_cov
end