# rule for "out" (univariate gradient case)
@rule UniSGP_Grad(:out, Marginalisation) (q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta,) = begin
    θ = mean(q_θ)
    μ_v = mean(q_v)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    Wg_bar = mean(q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ex = getEx(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    Ωx = Ex(μ_in)
    Ω1 = Cxθ_Xu(μ_in, θ, meta.Xu)
    Ku_mxu = (meta.KuuL * transpose(meta.KuuL)) \ mxu

    μ_ω = Ωx + Ω1 * (μ_v - Ku_mxu)

    return MvNormalMeanPrecision(μ_ω, Wg_bar)
end