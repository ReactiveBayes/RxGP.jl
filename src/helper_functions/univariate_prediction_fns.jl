export predict_GP

#---- Define prediction functions ----#

"""
    predict_GP(; m_in, q_v, q_θ, meta) → (means, covs)

Posterior predictive distribution at test inputs, per Eq. (30) of the dID-VSGP paper:

    μ_{f̃*} = L̃m(x*) + K̃_xu(x*)(μ_v - K_uu⁻¹ mᵤ)
    Σ_{f̃*} = K̃_xx(x*) + K̃_xu(x*)(Σ_v - K_uu⁻¹)K̃_xu(x*)ᵀ

The operator encoded in meta (`:fn`, `:grad`, `:joint_fn_grad`, or custom) determines the
output dimension P at each test point:

  - `:fn`            P=1     → means::Vector{Float64},   covs::Vector{Matrix{Float64}}
  - `:grad`          P=D     → same shapes, P-dim entries
  - `:joint_fn_grad` P=1+D   → same shapes, P-dim entries

Arguments
    m_in: vector of D-dimensional inputs, e.g. [[1.0, 2.0], [3.0, 4.0], ...]
    q_v::MultivariateNormalDistributionsFamily: variational posterior over inducing variables.
    q_θ::Any: kernel hyperparameters (PointMass or plain vector).
    meta::UniSGPMeta: meta object built by get_UniSGPMeta.

Returns
    (means, covs)
        means::Vector{<:AbstractVector}: P-dim predictive mean vector per test input.
        covs::Vector{<:AbstractMatrix}: P×P predictive covariance matrix per test input.
"""
function predict_GP(; m_in::AbstractVector{<:AbstractVector{<:Real}},
                     q_v::MultivariateNormalDistributionsFamily,
                     q_θ::Any,
                     meta::UniSGPMeta)
    θ = typeof(q_θ) <: Union{MultivariateNormalDistributionsFamily, PointMass} ? mean(q_θ) : q_θ
    μ_v, Σ_v = mean_cov(q_v)

    mf     = getMeanFn(meta)
    Xu     = getInducingInput(meta)
    Lm_fn  = getLm_fn(meta)
    Kxu_fn = getKxu_fn(meta)
    Kxx_fn = getKxx_fn(meta)

    mxu    = apply_mean_fn.(Xu, mf)
    Ku_mxu = meta.KuuF \ mxu        # K_uu⁻¹ mᵤ  (M-vector)
    Kuu_inv = meta.KuuF \ I          # K_uu⁻¹     (M×M)

    means = AbstractVector{Float64}[]
    covs  = Matrix{Float64}[]

    for x in m_in
        Lm  = Lm_fn(x)              # P-vector:  L̃m(x*)
        Kxu = Kxu_fn(x, θ, Xu)     # P×M:       K̃_xu(x*)
        Kxx = Kxx_fn(x, θ)         # P×P:       K̃_xx(x*)

        μ = Lm + Kxu * (μ_v - Ku_mxu)
        Σ = Kxx + Kxu * (Σ_v - Kuu_inv) * transpose(Kxu)

        push!(means, μ)
        push!(covs,  Symmetric(Σ))
    end

    return means, covs
end