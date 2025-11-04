# rule for "in" (univariate gradient case)
@rule UniSGP_Grad(:in, Marginalisation) (q_out::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta) = begin
    θ = mean(q_θ)
    μ_v, Σ_v = mean_cov(q_v)
    μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
    Wg_bar = mean(q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    Ku_mxu = (meta.KuuL * transpose(meta.KuuL)) \ mxu
    Dx = (x) -> Dxθ(x, θ)
    Cx = (x) -> Cxθ_Xu(x, θ, meta.Xu)
    CxT = (x) -> transpose(Cx(x))
    Qx = (x) -> Dx(x) - Cx(x) * ((meta.KuuL * transpose(meta.KuuL)) \ transpose(Cx(x)))
    Rv = μ_v * transpose(μ_v) + Σ_v
    WC = (x) -> Wg_bar * Cx(x)
    CTWC = (x) -> CxT(x) * Wg_bar * Cx(x)
    CTWC_Ku_mxu = (x) -> CTWC(x) * Ku_mxu

    part_A = (x) -> tr(Wg_bar * Qx(x))
    part_B = (x) -> - 2 * transpose(μ_ω) * Wg_bar * ( Ex(x) + Cx(x) * (μ_v - Ku_mxu) )
    part_C = (x) -> (
        tr(Rv * CTWC(x))
        + (transpose(Ku_mxu) - 2 * transpose(μ_v)) * CTWC_Ku_mxu(x)
        + 2 * transpose(Ex(x)) * WC(x) * (μ_v - Ku_mxu)
    )

    log_backwardmess = (x) -> -0.5 * (part_A(x) + part_B(x) + part_C(x))
    
    return get_dims_data(meta) < 2 ? ContinuousUnivariateLogPdf(log_backwardmess) : ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
end