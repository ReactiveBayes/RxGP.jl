# rule for "in" (univariate dID case)
@rule UniSGP_dID(:in, Marginalisation) (q_out::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta) = begin
    θ = mean(q_θ)
    μ_v, Σ_v = mean_cov(q_v)
    μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
    Wg_bar = mean(q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Lm_fn = getLm_fn(meta)
    Kxx_fn = getKxx_fn(meta)
    Kxu_fn = getKxu_fn(meta)

    Ku_mxu = meta.KuuF \ mxu
    Lx = (x) -> Lm_fn(x)
    Kx = (x) -> Kxu_fn(x, θ, meta.Xu)
    KxT = (x) -> transpose(Kx(x))
    Qx = (x) -> Kxx_fn(x, θ) - Kx(x) * (meta.KuuF \ transpose(Kx(x)))
    Rv = μ_v * transpose(μ_v) + Σ_v
    WK = (x) -> Wg_bar * Kx(x)
    KTWK = (x) -> KxT(x) * Wg_bar * Kx(x)
    KTWK_Ku_mxu = (x) -> KTWK(x) * Ku_mxu

    part_A = (x) -> tr(Wg_bar * Qx(x))
    part_B = (x) -> - 2 * transpose(μ_ω) * Wg_bar * ( Lx(x) + Kx(x) * (μ_v - Ku_mxu) )
    part_C = (x) -> (
        transpose(Lx(x)) * Wg_bar * Lx(x)
        + tr(Rv * KTWK(x))
        + (transpose(Ku_mxu) - 2 * transpose(μ_v)) * KTWK_Ku_mxu(x)
        + 2 * transpose(Lx(x)) * WK(x) * (μ_v - Ku_mxu)
    )

    log_backwardmess = (x) -> -0.5 * (part_A(x) + part_B(x) + part_C(x))
    
    return get_dims_data(meta) == 1 ? ContinuousUnivariateLogPdf(log_backwardmess) : ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
end