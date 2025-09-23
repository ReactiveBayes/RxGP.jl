# rule for "out" edge (multivariate case)

@rule MultiSGP(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_v = mean(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    cache = getGPCache(meta)
    Ψ1_trans = getΨ1_trans(meta)
    M = length(Xu) #number of inducing points
    D = size(W,1)
    method = getmethod(meta)
    @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
    approximate_kernel_expectation!(Ψ1_trans,method, (x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]), q_in)
    return MvNormalMeanPrecision(map!(yi -> jdotavx(Ψ1_trans, yi),getcache(cache, (:μ_y, D)), μ_v), W)
end

@rule MultiSGP(:out, Marginalisation) (q_in::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,q_θ::PointMass,meta::MultiSGPMeta,) = begin
    μ_v = mean(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    cache = getGPCache(meta)
    Ψ1_trans = getΨ1_trans(meta)
    M = length(Xu) #number of inducing points
    D = size(W,1)
    method = getmethod(meta)
    @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
    approximate_kernel_expectation!(Ψ1_trans,method, (x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ),Xu, [x]), q_in) 
    return MvNormalMeanPrecision(map!(yi -> jdotavx(Ψ1_trans, yi),getcache(cache, (:μ_y, D)), μ_v), W)
end

@rule MultiSGP(:out, Marginalisation) (q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_v = mean(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    cache = getGPCache(meta)
    Ψ1_trans = getΨ1_trans(meta)
    M = length(Xu) #number of inducing points
    D = size(W,1)
    method = getmethod(meta)
    @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
    kernelmatrix!(Ψ1_trans, kernel(θ),Xu, [mean(q_in)])
    return MvNormalMeanPrecision(map!(yi -> jdotavx(Ψ1_trans, yi),getcache(cache, (:μ_y, D)), μ_v), W)
end

