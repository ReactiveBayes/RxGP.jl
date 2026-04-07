# rule for "v" edge (univariate gradient batch case)
@rule UniSGP_Grad_Batch(:v, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta) = begin
    θ = mean(q_θ)
    Wg_bar = mean(q_Wg); @assert maximum(abs, Wg_bar - transpose(Wg_bar)) < 1e-8 "Wg_bar is not symmetric"
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    D = get_dims_data(meta)
    M = length(meta.Xu)
    Ex = getEx(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    μ_ω_batch, _ = mean_cov_vector_matrix(q_out)
    μ_in_flat = mean(q_in)
    N_j = div(length(μ_ω_batch), D)
    is_dist_in = q_in isa Distribution
    Σ_in_flat = is_dist_in ? cov(q_in) : nothing

    Ω3_sum = zeros(M, M)
    ξ_accum = zeros(M)

    for j in 1:N_j
        idx = (j-1)*D+1 : j*D
        μ_ω_j = μ_ω_batch[idx]

        if is_dist_in
            μ_j = μ_in_flat[idx]; Σ_j = Σ_in_flat[idx, idx]
            q_j = D == 1 ? NormalMeanVariance(μ_j[1], Σ_j[1,1]) : MvNormalMeanCovariance(μ_j, Matrix(Σ_j))
            Ω1 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu), q_j)
            Ω3 = approximate_kernel_expectation(meta.method, (x) -> transpose(Cxθ_Xu(x, θ, meta.Xu))*Wg_bar*Cxθ_Xu(x, θ, meta.Xu), q_j)
            Ω4 = approximate_kernel_expectation(meta.method, (x) -> transpose(Ex(x))*Wg_bar*Cxθ_Xu(x, θ, meta.Xu), q_j)
        else
            μ_in = μ_in_flat[idx]
            Ω1 = Cxθ_Xu(μ_in, θ, meta.Xu)
            Ω3 = transpose(Ω1) * Wg_bar * Ω1
            Ω4 = transpose(Ex(μ_in)) * Wg_bar * Ω1
        end

        Ω3_sum += Ω3
        ξ_accum += vec(Ω3 * Ku_mxu + transpose(Ω1) * Wg_bar * μ_ω_j - transpose(Ω4))
    end

    W_v = Symmetric(Ω3_sum + 1e-8I, :U)
    return MvNormalWeightedMeanPrecision(ξ_accum, W_v)
end
