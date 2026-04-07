# rule for "in" (univariate gradient batch case)
@rule UniSGP_Grad_Batch(:in, Marginalisation) (q_out::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta) = begin
    θ = mean(q_θ)
    μ_v, Σ_v = mean_cov(q_v)
    Wg_bar = mean(q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    Ku_mxu = meta.KuuF \ mxu
    D = get_dims_data(meta)
    Rv = μ_v * transpose(μ_v) + Σ_v

    μ_ω_batch, _ = mean_cov_vector_matrix(q_out)
    N_j = div(length(μ_ω_batch), D)

    Dx = (x) -> Dxθ(x, θ)
    Cx = (x) -> Cxθ_Xu(x, θ, meta.Xu)
    CxT = (x) -> transpose(Cx(x))
    Qx = (x) -> Dx(x) - Cx(x) * (meta.KuuF \ transpose(Cx(x)))
    WC = (x) -> Wg_bar * Cx(x)
    CTWC = (x) -> CxT(x) * Wg_bar * Cx(x)
    CTWC_Ku_mxu = (x) -> CTWC(x) * Ku_mxu

    part_A = (x) -> tr(Wg_bar * Qx(x))
    part_B = (x, μ_ω_j) -> - 2 * transpose(μ_ω_j) * Wg_bar * ( Ex(x) + Cx(x) * (μ_v - Ku_mxu) )
    part_C = (x) -> (
        transpose(Ex(x)) * Wg_bar * Ex(x)
        + tr(Rv * CTWC(x))
        + (transpose(Ku_mxu) - 2 * transpose(μ_v)) * CTWC_Ku_mxu(x)
        + 2 * transpose(Ex(x)) * WC(x) * (μ_v - Ku_mxu)
    )

    log_backwardmess = (x_batch) -> begin
        s = 0.0
        for j in 1:N_j
            x_j = D == 1 ? x_batch[(j-1)*D+1] : x_batch[(j-1)*D+1 : j*D]
            μ_ω_j = μ_ω_batch[(j-1)*D+1 : j*D]
            s += -0.5 * (part_A(x_j) + part_B(x_j, μ_ω_j) + part_C(x_j))
        end
        return s
    end

    return ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
end
