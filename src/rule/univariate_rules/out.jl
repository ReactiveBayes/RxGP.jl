# rule for "out" (univariate case)

# GP inputs are random variables (general case)
@rule UniSGP(:out, Marginalisation) (q_in::UnivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_w::Any, q_θ::PointMass, meta::UniSGPMeta,) = begin
    kernel = getKernel(meta)
    θ = mean(q_θ) 
    μ_v = mean(q_v)
    Ψ1_trans = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu,[x]) , q_in)
    return NormalMeanPrecision(jdotavx(Ψ1_trans ,μ_v), mean(q_w))
end

# GP inputs are known
@rule UniSGP(:out, Marginalisation) (q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::Any, q_θ::PointMass, meta::UniSGPMeta,) = begin
    kernel = getKernel(meta)
    θ = mean(q_θ)
    μ_v = mean(q_v)
    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(meta.Ψ1_trans,kernel(θ),meta.Xu, [mean(q_in)])
    return NormalMeanPrecision(jdotavx(meta.Ψ1_trans, μ_v), mean(q_w))
end

