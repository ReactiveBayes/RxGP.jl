# This file defines the Sparse Gaussian Process (SGP) node for univariate case
# In particular, y = f(x), where y ∈ ℝ is a scalar, and x ∈ ℝᴰ

export UniSGP_dID

"""
    UniSGP_dID

Decoupled inter-domain variational sparse Gaussian process (dID-VSGP) factor
node. Generalises [`UniSGP`](@ref) by applying an arbitrary deterministic
linear operator ``\\mathcal{L}`` to the latent GP, so that observations live in
the transformed space ``\\tilde{f}(x) = \\mathcal{L} f(x) \\in \\mathbb{R}^P``.

The composite node function is:

```math
\\tilde{\\phi}(\\tilde{\\mathbf{y}}, x, \\mathbf{v}, W, \\boldsymbol{\\theta})
= \\exp\\!\\Bigl(-\\tfrac{1}{2}\\operatorname{tr}\\bigl(W\\,\\tilde{A}_u(x, \\boldsymbol{\\theta})\\bigr)\\Bigr)\\;
  \\mathcal{N}\\!\\bigl(\\tilde{\\mathbf{y}} \\mid \\tilde{b}_u(x, \\mathbf{v}, \\boldsymbol{\\theta}),\\; W^{-1}\\bigr)
```

where ``\\tilde{b}_u = \\tilde{m}(x) + \\tilde{K}_{xu}(\\mathbf{v} - K_{uu}^{-1}\\mathbf{m}_u)``
and ``\\tilde{A}_u = \\tilde{K}_{xx'} - \\tilde{K}_{xu} K_{uu}^{-1} \\tilde{K}_{ux}``
with transformed kernels ``\\tilde{K}_{xu} = \\mathcal{L}_1 k(x, X_u)``, etc.
Inducing variables ``\\mathbf{v}`` remain in the latent (untransformed) space.

**Edges:** `[out, in, v, Wg, θ]` — output, input, transformed inducing variable, Wishart noise precision, kernel hyperparameters.
"""
struct UniSGP_dID end 

@node UniSGP_dID Stochastic [ out, in, v , Wg, θ] # out: output, in: input,  v: transformed-inducing points Kuu_inv * u , Wg: precision of process noise 

#---- Define average energy ----#
@average_energy UniSGP_dID (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta,) = begin
    θ = mean(q_θ)
    μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Wg_bar = mean(q_Wg)
    E_logWg = mean(logdet, q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    D = length(μ_ω)
    Rv = μ_v * transpose(μ_v) + Σ_v
    Lm_fn = getLm_fn(meta)
    Kxx_fn = getKxx_fn(meta)
    Kxu_fn = getKxu_fn(meta)

    if q_in isa Distribution
        Ωx = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x), q_in)
        Ω0 = approximate_kernel_expectation(meta.method, (x) -> Kxx_fn(x, θ), q_in)
        Ω1 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu), q_in)
        Ω2 = approximate_kernel_expectation(meta.method, (x) -> transpose(Kxu_fn(x, θ, meta.Xu))*Kxu_fn(x, θ, meta.Xu), q_in)
        Ω5 = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x) * transpose(Lm_fn(x)), q_in)
        Ω6 = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x) * transpose(μ_v) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
        Ω7 = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x) * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
        Ω8 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * μ_v * transpose(Lm_fn(x)), q_in)
        Ω9 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * Rv * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
        Ω10 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * μ_v * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
        Ω11 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * Ku_mxu * transpose(Lm_fn(x)), q_in)
        Ω12 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * Ku_mxu * transpose(μ_v) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
        Ω13 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
    else
        Ωx = Lm_fn(μ_in)
        Ω0 = Kxx_fn(μ_in, θ)
        Ω1 = Kxu_fn(μ_in, θ, meta.Xu)
        Ω2 = transpose(Ω1) * Ω1
        Ω5 = Ωx * transpose(Ωx)
        Ω6 = Ωx * transpose(μ_v) * transpose(Ω1)
        Ω7 = Ωx * transpose(Ku_mxu) * transpose(Ω1)
        Ω8 = Ω1 * μ_v * transpose(Ωx)
        Ω9 = Ω1 * Rv * transpose(Ω1)
        Ω10 = Ω1 * μ_v * transpose(Ku_mxu) * transpose(Ω1)
        Ω11 = Ω1 * Ku_mxu * transpose(Ωx)
        Ω12 = Ω1 * Ku_mxu * transpose(μ_v) * transpose(Ω1)
        Ω13 = Ω1 * Ku_mxu * transpose(Ku_mxu) * transpose(Ω1)
    end

    G1 = Ω0 - Ω1 * (meta.KuuF \ transpose(Ω1))
    A_G2 = typeof(q_out) <: MultivariateNormalDistributionsFamily ? μ_ω * transpose(μ_ω) + Σ_ω : μ_ω * transpose(μ_ω)
    B_G2 = μ_ω * ( transpose(Ωx) + ( transpose(μ_v) - transpose(Ku_mxu) ) * transpose(Ω1) )
    C_G2 = (Ωx + Ω1 * ( μ_v - Ku_mxu )) * transpose(μ_ω)
    D_G2 = Ω5 + Ω6 - Ω7 + Ω8 + Ω9 - Ω10 - Ω11 - Ω12 + Ω13
    G2 = A_G2 - B_G2 - C_G2 + D_G2

    return 0.5 * tr( Wg_bar * ( G1 + G2 ) ) + (D/2) * log(2 * π) - 0.5 * E_logWg
end