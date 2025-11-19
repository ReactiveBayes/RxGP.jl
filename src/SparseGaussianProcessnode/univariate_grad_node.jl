# This file defines the Sparse Gaussian Process (SGP) node for univariate case
# In particular, y = f(x), where y ∈ ℝ is a scalar, and x ∈ ℝᴰ

export UniSGP_Grad

struct UniSGP_Grad end 

@node UniSGP_Grad Stochastic [ out, in, v , Wg, θ] # out: output, in: input,  v: transformed-inducing points Kuu_inv * u , Wg: precision of process noise 

#---- Define average energy ----#
@average_energy UniSGP_Grad (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta,) = begin
    θ = mean(q_θ)
    μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Wg_bar = mean(q_Wg)
    E_logWg = mean(logdet, q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    D = length(μ_ω)
    Rv = μ_v * transpose(μ_v) + Σ_v
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    Ku_mxu = meta.KuuF \ mxu
    Ωx = Ex(μ_in)
    Ω0 = Dxθ(μ_in, θ)
    Ω1 = Cxθ_Xu(μ_in, θ, meta.Xu)
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

    G1 = Ω0 - Ω1 * (meta.KuuF \ transpose(Ω1))
    A_G2 = typeof(q_out) <: MultivariateNormalDistributionsFamily ? μ_ω * transpose(μ_ω) + Σ_ω : μ_ω * transpose(μ_ω)
    B_G2 = μ_ω * ( transpose(Ωx) + ( transpose(μ_v) - transpose(Ku_mxu) ) * transpose(Ω1) )
    C_G2 = (Ωx + Ω1 * ( μ_v - Ku_mxu )) * transpose(μ_ω)
    D_G2 = Ω5 + Ω6 - Ω7 + Ω8 + Ω9 - Ω10 - Ω11 - Ω12 + Ω13
    G2 = A_G2 - B_G2 - C_G2 + D_G2

    return 0.5 * tr( Wg_bar * ( G1 + G2 ) ) + (D/2) * log(2 * π) - 0.5 * E_logWg
end