# rule for "θ" edge (univariate batch case)
@rule UniSGP_Batch(:θ, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_w::NOISE_w, meta::UniSGPMeta,) = begin
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w_bar = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    KuuF = (θ) -> fastcholesky(kernelmatrix(kernel(θ), Xu))
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = (θ) -> KuuF(θ) \ mxu
    mxuT_KuT = (θ) -> transpose(Ku_mxu(θ))
    D = get_dims_data(meta)

    μ_y_batch, _ = mean_cov_vector_matrix(q_out)
    N_j = length(μ_y_batch)
    μ_in_flat = mean(q_in)
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
            if is_dist_in
                q_j = q_points[j]
                Ψ0 = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), [x], [x]), q_j)[1]
                Ψ1 = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), [x], Xu), q_j)
                Ψ2 = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_j)
                Ψ3 = approximate_kernel_expectation(meta.method, (x) -> apply_mean_fn(x, mf) * kernelmatrix(kernel(θ), [x], Xu), q_j)
            else
                x_j = x_points[j]
                Ψ0 = kernelmatrix(kernel(θ), [x_j], [x_j])[1]
                Ψ1 = kernelmatrix(kernel(θ), [x_j], Xu)
                Ψ2 = kernelmatrix(kernel(θ), Xu, [x_j]) * kernelmatrix(kernel(θ), [x_j], Xu)
                Ψ3 = apply_mean_fn(x_j, mf) * kernelmatrix(kernel(θ), [x_j], Xu)
            end

            Ψ2_Ku_mxu = Ψ2 * Ku_mxu(θ)

            I1_θ = Ψ0 - tr( KuuF(θ) \ Ψ2 )
            I5_θ = (
                - 2 * μ_y_batch[j] * jdotavx(Ψ1, ( μ_v - Ku_mxu(θ) ))
                + tr( Rv * Ψ2 )
                + jdotavx(mxuT_KuT(θ), Ψ2_Ku_mxu)
                + 2 * jdotavx(Ψ3, (μ_v - Ku_mxu(θ)))
                - 2 * jdotavx(transpose(μ_v), Ψ2_Ku_mxu)
            )
            s += I1_θ + I5_θ
        end
        return -0.5 * w_bar * s
    end

    return get_dims_theta(meta) < 2 ? ContinuousUnivariateLogPdf(log_backwardmess) : ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
end
