# rule for "out" (univariate gradient batch case)
@rule UniSGP_Grad_Batch(:out, Marginalisation) (q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta,) = begin
    θ = mean(q_θ)
    μ_v = mean(q_v)
    Wg_bar = mean(q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    D = get_dims_data(meta)
    Ex = getEx(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    μ_in_flat = mean(q_in)
    N_j = div(length(μ_in_flat), D)
    is_dist_in = q_in isa Distribution
    Σ_in_flat = is_dist_in ? cov(q_in) : nothing

    μ_ω_batch = zeros(D * N_j)
    for j in 1:N_j
        idx = (j-1)*D+1 : j*D

        if is_dist_in
            μ_j = μ_in_flat[idx]; Σ_j = Σ_in_flat[idx, idx]
            q_j = D == 1 ? NormalMeanVariance(μ_j[1], Σ_j[1,1]) : MvNormalMeanCovariance(μ_j, Matrix(Σ_j))
            Ωx = approximate_kernel_expectation(meta.method, (x) -> Ex(x), q_j)
            Ω1 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu), q_j)
        else
            μ_in = μ_in_flat[idx]
            Ωx = Ex(μ_in)
            Ω1 = Cxθ_Xu(μ_in, θ, meta.Xu)
        end

        μ_ω_batch[idx] = Ωx + Ω1 * (μ_v - Ku_mxu)
    end

    # Block-diagonal precision: kron(I_{N_j}, Wg_bar)
    W_out = zeros(D * N_j, D * N_j)
    for j in 1:N_j
        idx = (j-1)*D+1 : j*D
        W_out[idx, idx] = Wg_bar
    end

    return MvNormalMeanPrecision(μ_ω_batch, W_out)
end
