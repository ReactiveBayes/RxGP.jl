# rule for "in" (univariate batch case)
@rule UniSGP_Batch(:in, Marginalisation) (q_out::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_w::NOISE_w, q_θ::PointMass, meta::UniSGPMeta) = begin
    w_bar = mean(q_w)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    μ_v, Σ_v = mean_cov(q_v)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Uv = fastcholesky(Σ_v + μ_v * μ_v').U
    Ku_mxu = meta.KuuF \ mxu
    D = get_dims_data(meta)

    μ_y_batch, _ = mean_cov_vector_matrix(q_out)
    N_j = length(μ_y_batch)

    # Per-point backward message closures (shared across points)
    B_trans = (x) -> kernelmatrix(kernel(θ), meta.Xu, [x])
    α_fn = (x) -> meta.KuuF.L \ B_trans(x)
    A = (x) -> kernelmatrix(kernel(θ), [x]) .- dot(α_fn(x), α_fn(x))
    β = (x) -> Uv * B_trans(x)
    mx = (x) -> apply_mean_fn(x, mf)
    mxuT_KuuInvT_BT = (x) -> dot(Ku_mxu, B_trans(x))
    B_KuuInv_mxu = (x) -> dot(B_trans(x), Ku_mxu)

    partA = (x) -> -0.5 * w_bar * A(x)[1]
    partB = (x, y_j) -> w_bar * y_j * (
        mx(x)
        + dot(B_trans(x), μ_v)
        - B_KuuInv_mxu(x)
    )
    partC = (x) -> -0.5 * w_bar * (
        mx(x)^2
        + dot(β(x), β(x))
        + mxuT_KuuInvT_BT(x)*B_KuuInv_mxu(x)
        + 2*mx(x)*dot(B_trans(x), μ_v)
        - 2*mx(x)*B_KuuInv_mxu(x)
        - 2*dot(B_trans(x), μ_v)*B_KuuInv_mxu(x)
    )

    log_backwardmess = (x_batch) -> begin
        s = 0.0
        for j in 1:N_j
            x_j = D == 1 ? x_batch[(j-1)*D+1] : x_batch[(j-1)*D+1 : j*D]
            s += partA(x_j) + partB(x_j, μ_y_batch[j]) + partC(x_j)
        end
        return s
    end

    return ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
end
