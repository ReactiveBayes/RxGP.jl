# rule for "θ" edge (univariate gradient batch case)
@rule UniSGP_Grad_Batch(:θ, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, meta::UniSGPMeta,) = begin
    μ_v, Σ_v = mean_cov(q_v)
    Rv = μ_v * transpose(μ_v) + Σ_v
    Wg_bar = mean(q_Wg)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    KuuF = (θ) -> fastcholesky(kernelmatrix(kernel(θ), Xu))
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    mxuT_KuT = transpose(Ku_mxu)
    D = get_dims_data(meta)
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    μ_ω_batch, _ = mean_cov_vector_matrix(q_out)
    μ_in_flat = mean(q_in)
    N_j = div(length(μ_ω_batch), D)
    is_dist_in = q_in isa Distribution

    # Pre-extract per-point data
    if is_dist_in
        Σ_in_flat = cov(q_in)
        q_points = [let idx = (j-1)*D+1:j*D
            μ_j = μ_in_flat[idx]; Σ_j = Σ_in_flat[idx, idx]
            D == 1 ? NormalMeanVariance(μ_j[1], Σ_j[1,1]) : MvNormalMeanCovariance(μ_j, Matrix(Σ_j))
        end for j in 1:N_j]
    else
        x_points = [μ_in_flat[(j-1)*D+1:j*D] for j in 1:N_j]
    end

    log_backwardmess = (θ) -> begin
        s = 0.0
        for j in 1:N_j
            idx_out = (j-1)*D+1 : j*D
            μ_ω_j = μ_ω_batch[idx_out]

            if is_dist_in
                q_j = q_points[j]
                Ω0 = approximate_kernel_expectation(meta.method, (x) -> Dxθ(x, θ), q_j)
                Ω1 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu), q_j)
                Ω3 = approximate_kernel_expectation(meta.method, (x) -> transpose(Cxθ_Xu(x, θ, meta.Xu))*Wg_bar*Cxθ_Xu(x, θ, meta.Xu), q_j)
                Ω4 = approximate_kernel_expectation(meta.method, (x) -> transpose(Ex(x))*Wg_bar*Cxθ_Xu(x, θ, meta.Xu), q_j)
            else
                x_j = x_points[j]
                Ω1_local = Cxθ_Xu(x_j, θ, meta.Xu)
                Ω0 = Dxθ(x_j, θ)
                Ω1 = Ω1_local
                Ω3 = transpose(Ω1_local) * Wg_bar * Ω1_local
                Ω4 = transpose(Ex(x_j)) * Wg_bar * Ω1_local
            end

            Ku_mxu_θ = KuuF(θ) \ mxu
            G1 = Ω0 - Ω1 * (KuuF(θ) \ transpose(Ω1))
            part_A = 2 * dot(Ω4, (μ_v - Ku_mxu_θ)) + dot((mxuT_KuT - 2*transpose(μ_v)), Ω3*Ku_mxu_θ) + tr(Ω3 * Rv)
            part_B = 2 * dot(transpose(μ_ω_j), Wg_bar * Ω1 * (μ_v - Ku_mxu_θ))

            s += -0.5 * tr( Wg_bar * G1 ) - 0.5 * (part_A - part_B)
        end
        return s
    end

    return get_dims_theta(meta) == 1 ? ContinuousUnivariateLogPdf(log_backwardmess) : ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
end
