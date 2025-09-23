# rule for "v" edge (multivariate case)
@rule MultiSGP(:v, Marginalisation) (q_out::MultivariateGaussianDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_w::Any,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    W = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    cache = getGPCache(meta)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension
    method = getmethod(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ),Xu, [x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu) + 1e-12*I, q_in)

    W_v = getcache(cache, (:W_v, (D*M,D*M)))
    kron!(W_v,W, Ψ2) # precision matrix 
    return MvNormalWeightedMeanPrecision(vcat(Ψ1_trans .* mul_A_B!(cache,μ_y',W,1,D)...), W_v)
end

@rule MultiSGP(:v, Marginalisation) (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_w::Any,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    W = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    cache = getGPCache(meta)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension
    method = getmethod(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu,[x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu)+ 1e-12*I, q_in)

    W_v = getcache(cache, (:W_v, (D*M,D*M)))
    kron!(W_v,W, Ψ2)
    return MvNormalWeightedMeanPrecision(vcat(Ψ1_trans .* mul_A_B!(cache,μ_y',W,1,D)...), W_v)
end

@rule MultiSGP(:v, Marginalisation) (q_out::PointMass, q_in::PointMass, q_w::Any,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    W = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    cache = getGPCache(meta)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    M = length(Xu) #number of inducing points
    D = length(μ_y) #dimension
    method = getmethod(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 

    kernelmatrix!(Ψ1_trans,kernel(θ),Xu, [mean(q_in)])
    Ψ2 =  kernelmatrix(kernel(θ), Xu, [mean(q_in)]) * kernelmatrix(kernel(θ), [mean(q_in)], Xu) + 1e-12*I

    W_v = getcache(cache, (:W_v, (D*M,D*M)))
    kron!(W_v,W, Ψ2) # precision matrix 
    return MvNormalWeightedMeanPrecision(vcat(Ψ1_trans .* mul_A_B!(cache,μ_y',W,1,D)...), W_v)
end
