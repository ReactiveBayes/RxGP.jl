# rule for "θ" edge (univariate case)
@rule UniSGP(:θ, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_w::NOISE_w, meta::UniSGPMeta,) = begin
    μ_y, Σ_y = mean_cov_scalar_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu))

    mf = getMeanFn(meta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(meta.Xu, mf)
    mxuT_KuT = (θ) -> transpose(mxu) * transpose(Kuu_inverse(θ))
    Ku_mxu = (θ) -> Kuu_inverse(θ) * mxu

    Ψ_0 = (θ) -> kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
    Ψ_1 = (θ) -> kernelmatrix(kernel(θ), [μ_in], Xu) 
    Ψ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu)
    I1_θ = (θ) -> Ψ_0(θ) - tr( Kuu_inverse(θ) * Ψ_2(θ) )
    I5_θ = (θ) -> (
        - 2 * μ_y * Ψ_1(θ) * ( μ_v - Ku_mxu(θ) )
        + tr( Rv * Ψ_2(θ) ) 
        + mxuT_KuT(θ) * Ψ_2(θ) * Ku_mxu(θ)
        + 2 * mx * Ψ_1(θ) * μ_v  
        - 2 * mx * Ψ_1(θ) * Ku_mxu(θ) 
        - 2 * transpose(μ_v) * Ψ_2(θ) * Ku_mxu(θ)
        )
    log_backwardmess = (θ) -> -0.5 * w * (I1_θ(θ) + I5_θ(θ))
    
    # return project_MvN_fn(log_backwardmess, get_dims_theta(meta))
    return get_dims_theta(meta) < 2 ? ContinuousUnivariateLogPdf(log_backwardmess) : ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
end