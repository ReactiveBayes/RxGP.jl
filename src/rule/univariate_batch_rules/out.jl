# rule for "out" (univariate batch case)
@rule UniSGP_Batch(:out, Marginalisation) (q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_w::Any, q_θ::PointMass, meta::UniSGPMeta,) = begin
    kernel = getKernel(meta)
    θ = mean(q_θ)
    μ_v = mean(q_v)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    D = get_dims_data(meta)

    μ_in_flat = mean(q_in)
    N_j = div(length(μ_in_flat), D)
    is_dist_in = q_in isa Distribution
    Σ_in_flat = is_dist_in ? cov(q_in) : nothing

    μ_y_batch = zeros(N_j)
    for j in 1:N_j
        idx = (j-1)*D+1 : j*D

        if is_dist_in
            μ_j = μ_in_flat[idx]; Σ_j = Σ_in_flat[idx, idx]
            q_j = D == 1 ? NormalMeanVariance(μ_j[1], Σ_j[1,1]) : MvNormalMeanCovariance(μ_j, Matrix(Σ_j))
            Ψx = approximate_kernel_expectation(meta.method, (x) -> [apply_mean_fn(x, mf)], q_j)[1]
            Ψ1_trans = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_j)
        else
            x_j = μ_in_flat[idx]
            Ψx = apply_mean_fn(x_j, mf)
            Ψ1_trans = kernelmatrix(kernel(θ), meta.Xu, [x_j])
        end

        μ_y_batch[j] = Ψx + jdotavx(Ψ1_trans, μ_v) - jdotavx(Ψ1_trans, Ku_mxu)
    end

    w_bar = mean(q_w)
    return MvNormalMeanPrecision(μ_y_batch, w_bar * Diagonal(ones(N_j)))
end
