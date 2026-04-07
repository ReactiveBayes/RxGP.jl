# This file defines the Batch Sparse Gaussian Process (SGP) node for univariate gradient case
# ω_j = ∇f(x_j), j=1..N_j, where ω_j ∈ ℝᴰ and x_j ∈ ℝᴰ
# out ∈ ℝ^{D·N_j} (stacked gradient outputs), in ∈ ℝ^{D·N_j} (stacked inputs)

export UniSGP_Grad_Batch

struct UniSGP_Grad_Batch end

@node UniSGP_Grad_Batch Stochastic [out, in, v, Wg, θ]

#---- Define average energy ----#
@average_energy UniSGP_Grad_Batch (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta,) = begin
    θ = mean(q_θ)
    μ_v, Σ_v = mean_cov(q_v)
    Wg_bar = mean(q_Wg)
    E_logWg = mean(logdet, q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    D = get_dims_data(meta)
    Rv = μ_v * transpose(μ_v) + Σ_v
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    μ_ω_batch, Σ_ω_batch = mean_cov_vector_matrix(q_out)
    μ_in_flat = mean(q_in)
    N_j = div(length(μ_ω_batch), D)
    is_dist_in = q_in isa Distribution
    Σ_in_flat = is_dist_in ? cov(q_in) : nothing

    G_sum = zeros(D, D)
    for j in 1:N_j
        idx = (j-1)*D+1 : j*D
        μ_ω = μ_ω_batch[idx]
        Σ_ω = Σ_ω_batch[idx, idx]

        if is_dist_in
            μ_j = μ_in_flat[idx]; Σ_j = Σ_in_flat[idx, idx]
            q_j = D == 1 ? NormalMeanVariance(μ_j[1], Σ_j[1,1]) : MvNormalMeanCovariance(μ_j, Matrix(Σ_j))
            Ωx = approximate_kernel_expectation(meta.method, (x) -> Ex(x), q_j)
            Ω0 = approximate_kernel_expectation(meta.method, (x) -> Dxθ(x, θ), q_j)
            Ω1 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu), q_j)
            Ω2 = approximate_kernel_expectation(meta.method, (x) -> transpose(Cxθ_Xu(x, θ, meta.Xu))*Cxθ_Xu(x, θ, meta.Xu), q_j)
            Ω5 = approximate_kernel_expectation(meta.method, (x) -> Ex(x) * transpose(Ex(x)), q_j)
            Ω6 = approximate_kernel_expectation(meta.method, (x) -> Ex(x) * transpose(μ_v) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_j)
            Ω7 = approximate_kernel_expectation(meta.method, (x) -> Ex(x) * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_j)
            Ω8 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * μ_v * transpose(Ex(x)), q_j)
            Ω9 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * Rv * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_j)
            Ω10 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * μ_v * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_j)
            Ω11 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * Ku_mxu * transpose(Ex(x)), q_j)
            Ω12 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * Ku_mxu * transpose(μ_v) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_j)
            Ω13 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_j)
        else
            μ_in = μ_in_flat[idx]
            Ωx = Ex(μ_in)
            Ω0 = Dxθ(μ_in, θ)
            Ω1 = Cxθ_Xu(μ_in, θ, meta.Xu)
            Ω2 = transpose(Ω1) * Ω1
            Ω5 = Ωx * transpose(Ωx)
            Ω6 = Ωx * transpose(μ_v) * transpose(Ω1)
            Ω7 = Ωx * transpose(Ku_mxu) * transpose(Ω1)
            Ω8 = Ω1 * μ_v * transpose(Ωx)
            Ω9 = Ω1 * Rv * transpose(Ω1)
            Ω10 = Ω1 * μ_v * transpose(Ku_mxu) * transpose(Ω1)
            Ω11 = Ω1 * Ku_mxu * transpose(Ωx)
            Ω12 = Ω1 * Ku_mxu * transpose(μ_v) * transpose(Ω1)
            Ω13 = Ω1 * Ku_mxu * transpose(Ku_mxu) * transpose(Ω1)
        end

        G1 = Ω0 - Ω1 * (meta.KuuF \ transpose(Ω1))
        A_G2 = μ_ω * transpose(μ_ω) + Σ_ω
        B_G2 = μ_ω * ( transpose(Ωx) + ( transpose(μ_v) - transpose(Ku_mxu) ) * transpose(Ω1) )
        C_G2 = (Ωx + Ω1 * ( μ_v - Ku_mxu )) * transpose(μ_ω)
        D_G2 = Ω5 + Ω6 - Ω7 + Ω8 + Ω9 - Ω10 - Ω11 - Ω12 + Ω13
        G2 = A_G2 - B_G2 - C_G2 + D_G2

        G_sum += G1 + G2
    end

    return 0.5 * tr( Wg_bar * G_sum ) + N_j*(D/2) * log(2 * π) - N_j*0.5 * E_logWg
end
