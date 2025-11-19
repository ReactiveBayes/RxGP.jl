# rule for "θ" edge (univariate gradient case)
@rule UniSGP_Grad(:θ, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, meta::UniSGPMeta,) = begin
    μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Wg_bar = mean(q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    Rv = μ_v * transpose(μ_v) + Σ_v
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    if q_in isa Distribution
        Ω0 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> Dxθ(x, θ), q_in) + 1e-8*I
        Ω1 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu), q_in) + 1e-8*I
        Ω2 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> transpose(Cxθ_Xu(x, θ, meta.Xu))*Cxθ_Xu(x, θ, meta.Xu), q_in) + 1e-8*I
        Ω3 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> transpose(Cxθ_Xu(x, θ, meta.Xu))*Wg_bar*Cxθ_Xu(x, θ, meta.Xu), q_in) + 1e-8*I
        Ω4 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> transpose(Ex(x))*Wg_bar*Cxθ_Xu(x, θ, meta.Xu), q_in) + 1e-8*I
    else
        Ω0 = (θ) -> Dxθ(μ_in, θ)
        Ω1 = (θ) -> Cxθ_Xu(μ_in, θ, meta.Xu)
        Ω2 = (θ) -> transpose(Ω1(θ)) * Ω1(θ)
        Ω3 = (θ) -> transpose(Ω1(θ)) * Wg_bar * Ω1(θ)
        Ω4 = (θ) -> transpose(Ex) * Wg_bar * Ω1(θ)
    end


    G1 = (θ) -> Ω0(θ) - Ω1(θ) * (meta.KuuF \ transpose(Ω1(θ)))
    part_A = (θ) -> 2 * dot(Ω4(θ), (μ_v - Ku_mxu)) + dot((mxuT_KuT - 2*transpose(μ_v)), Ω3(θ)*Ku_mxu) + tr(Ω3(θ) * Rv)
    part_B = (θ) -> 2 * dot(transpose(μ_ω), Wg_bar * Ω1(θ) * (μ_v - Ku_mxu))

    log_backwardmess = (θ) -> -0.5 * tr( Wg_bar * G1(θ) ) - 0.5 * (part_A(θ) - part_B(θ))

    return get_dims_theta(meta) < 2 ? ContinuousUnivariateLogPdf(log_backwardmess) : ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
    # return project_MvN_fn(log_backwardmess, get_dims_theta(meta))
end