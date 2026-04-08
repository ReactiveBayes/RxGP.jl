# rule for "out" (univariate dID case)
@rule UniSGP_dID(:out, Marginalisation) (q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta,) = begin
    θ = mean(q_θ)
    μ_v = mean(q_v)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    Wg_bar = mean(q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    Lm_fn = getLm_fn(meta)
    Kxu_fn = getKxu_fn(meta)

    if q_in isa Distribution
        Ωx = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x), q_in)
        Ω1 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu), q_in)
    else
        Ωx = Lm_fn(μ_in)
        Ω1 = Kxu_fn(μ_in, θ, meta.Xu)
    end

    μ_ω = Ωx + Ω1 * (μ_v - Ku_mxu)

    return meta.dims_data == 1 ? NormalMeanPrecision(only(μ_ω), only(Wg_bar)) : MvNormalMeanPrecision(μ_ω, Wg_bar)
end