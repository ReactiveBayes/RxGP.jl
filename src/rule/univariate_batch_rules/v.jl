# rule for "v" edge (univariate batch case)
@rule UniSGP_Batch(:v, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_w::NOISE_w, q_θ::PointMass, meta::UniSGPMeta) = begin
    w_bar = mean(q_w)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    D = get_dims_data(meta)
    M = length(meta.Xu)

    μ_y_batch, _ = mean_cov_vector_matrix(q_out)
    N_j = length(μ_y_batch)
    μ_in_flat = mean(q_in)
    is_dist_in = q_in isa Distribution
    Σ_in_flat = is_dist_in ? cov(q_in) : nothing

    W_v = zeros(M, M)
    rhs = zeros(M)

    for j in 1:N_j
        idx = (j-1)*D+1 : j*D

        if is_dist_in
            μ_j = μ_in_flat[idx]; Σ_j = Σ_in_flat[idx, idx]
            q_j = D == 1 ? NormalMeanVariance(μ_j[1], Σ_j[1,1]) : MvNormalMeanCovariance(μ_j, Matrix(Σ_j))
            Ψ1_trans = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_j)
            Ψ2 = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_j) + 1e-8*I
            Ψ3 = approximate_kernel_expectation(meta.method, (x) -> apply_mean_fn(x, mf) * kernelmatrix(kernel(θ), [x], meta.Xu), q_j)
        else
            x_j = μ_in_flat[idx]
            mx = apply_mean_fn(x_j, mf)
            Ψ1_trans = kernelmatrix(kernel(θ), meta.Xu, [x_j])
            Ψ2 = kernelmatrix(kernel(θ), meta.Xu, [x_j]) * kernelmatrix(kernel(θ), [x_j], meta.Xu) + 1e-8*I
            Ψ3 = mx * kernelmatrix(kernel(θ), [x_j], meta.Xu)
        end

        W_v += w_bar * Ψ2
        rhs += vec(w_bar * (μ_y_batch[j] * Ψ1_trans - transpose(Ψ3)))
    end

    ξ_v = rhs + W_v * Ku_mxu

    return MvNormalWeightedMeanPrecision(ξ_v, W_v)
end
