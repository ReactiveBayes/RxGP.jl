# rule for "θ" edge (univariate case)
@rule UniSGP(:θ, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_w::NOISE_w, meta::UniSGPMeta,) = begin
    μ_y, Σ_y = mean_cov_scalar_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w_bar = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    KuuF = (θ) -> fastcholesky(kernelmatrix(kernel(θ),Xu))

    mf = getMeanFn(meta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = (θ) -> KuuF(θ) \ mxu
    mxuT_KuT = (θ) -> transpose(Ku_mxu(θ))

    if q_in isa Distribution
        Ψ_0 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), [x], [x]), q_in)[1]
        Ψ_1 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), [x], Xu), q_in)
        Ψ_2 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in)
    else
        Ψ_0 = (θ) -> kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
        Ψ_1 = (θ) -> kernelmatrix(kernel(θ), [μ_in], Xu) 
        Ψ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu)
    end

    I1_θ = (θ) -> Ψ_0(θ) - tr( KuuF(θ) \ Ψ_2(θ) )
    I5_θ = (θ) -> (
        - 2 * μ_y * jdotavx(Ψ_1(θ), ( μ_v - Ku_mxu(θ) ))
        + tr( Rv * Ψ_2(θ) ) 
        + jdotavx(mxuT_KuT(θ), Ψ_2(θ) * Ku_mxu(θ))
        + 2 * mx * jdotavx(Ψ_1(θ), (μ_v - Ku_mxu(θ)))  
        - 2 * jdotavx(transpose(μ_v), Ψ_2(θ) * Ku_mxu(θ))
    )

    log_backwardmess = (θ) -> -0.5 * w_bar * (I1_θ(θ) + I5_θ(θ))
    
    return get_dims_theta(meta) < 2 ? ContinuousUnivariateLogPdf(log_backwardmess) : ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
end