# rule for "θ" edge (univariate case)

@rule UniSGP(:θ, Marginalisation) (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::Any, meta::UniSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu) )
    Ψ_0 = (θ) -> kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
    Ψ_1 = (θ) -> kernelmatrix(kernel(θ), [μ_in], Xu) 
    Ψ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu)
    log_backwardmess = (θ) -> w * μ_y * (Ψ_1(θ) * μ_v)[1] - 0.5 * w * (Ψ_0(θ) + tr(Ψ_2(θ)*(Rv - Kuu_inverse(θ))))
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule UniSGP(:θ, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::Any, meta::UniSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> inv(kernelmatrix(kernel(θ),Xu))
    Ψ_0 = (θ) -> kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
    Ψ_1 = (θ) -> kernelmatrix(kernel(θ), [μ_in], Xu) 
    Ψ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) 
    log_backwardmess = (θ) -> w * μ_y * (Ψ_1(θ) * μ_v)[1] - 0.5 * w * (Ψ_0(θ) + tr(Ψ_2(θ)*(Rv - Kuu_inverse(θ))))
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule UniSGP(:θ, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateNormalDistributionsFamily, q_v::MultivariateGaussianDistributionsFamily, q_w::Any, meta::UniSGPMeta) = begin 
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w = mean(q_w)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = (θ) -> inv(kernelmatrix(kernel(θ),Xu))
    Ψ_0 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ_1 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in) 
    Ψ_2 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) 
    log_backwardmess = (θ) -> w * μ_y * (Ψ_1(θ) * μ_v)[1] - 0.5 * w * (Ψ_0(θ) + tr(Ψ_2(θ)*(Rv - Kuu_inverse(θ))))
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end
