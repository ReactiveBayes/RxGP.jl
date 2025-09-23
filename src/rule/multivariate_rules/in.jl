# rule for "in" edge (multivariate case)
@rule MultiSGP(:in, Marginalisation) (q_out::MultivariateNormalDistributionsFamily, q_v::MultivariateNormalDistributionsFamily,q_w::Any,q_θ::PointMass, meta::MultiSGPMeta) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = getKuuInverse(meta)
    cache = getGPCache(meta)
    M = length(Xu)
    D = length(μ_y)

    Ψ0 = (x) -> kernelmatrix(kernel(θ),[x])[1]
    Ψ1_trans = (x) -> kernelmatrix(kernel(θ),Xu,[x])
    Ψ2 = (x) -> Ψ1_trans(x) * Ψ1_trans(x)'
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W)

    log_backwardmess = (x) -> -0.5 * tr(W) * (Ψ0(x) - sum(Kuu_inverse .* Ψ2(x))) + sum(sumdiagV .* Ψ1_trans(x)) - 0.5 * sum(Ψ2(x) .* sumRvblk_W)
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule MultiSGP(:in, Marginalisation) (q_out::PointMass, q_v::MultivariateNormalDistributionsFamily,q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    logdetW = log(det(W))
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = getKuuInverse(meta)
    cache = getGPCache(meta)
    M = length(Xu)
    D = length(μ_y)

    Ψ0 = (x) -> kernelmatrix(kernel(θ),[x])[1]
    Ψ1_trans = (x) -> kernelmatrix(kernel(θ),Xu,[x])
    Ψ2 = (x) -> Ψ1_trans(x) * Ψ1_trans(x)'
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W)
    log_backwardmess = (x) -> -0.5 * tr(W) * (Ψ0(x) - sum(Kuu_inverse .* Ψ2(x))) + sum(sumdiagV .* Ψ1_trans(x)) - 0.5 * sum(Ψ2(x) .* sumRvblk_W) #- D/2 * log(2π) + 0.5*logdetW - 0.5*μ_y'*W*μ_y
    return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
end

@rule MultiSGP(:in, Marginalisation) (q_out::PointMass,q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateGaussianDistributionsFamily, q_w::Any,q_θ::PointMass, meta::MultiSGPMeta) = begin 
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W = mean(q_w) 
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Xu = getInducingInput(meta)
    Kuu_inverse = getKuuInverse(meta)
    M = length(Xu)
    D = length(μ_y)
    
    Ψ0 = (x) -> kernelmatrix(kernel(θ),[x])[1]
    Ψ1_trans = (x) -> kernelmatrix(kernel(θ),Xu,[x])
    Ψ2 = (x) -> Ψ1_trans(x) * Ψ1_trans(x)'
    Rv = Σ_v + μ_v * μ_v'
    V = μ_v * μ_y' * W
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Rv,D,M) .* W)

    neg_log_backwardmess = (x) -> -(-0.5 * tr(W) * (Ψ0(x) - sum(Kuu_inverse .* Ψ2(x))) + sum(sumdiagV .* Ψ1_trans(x)) - 0.5 * sum(Ψ2(x) .* sumRvblk_W))
    grad_func! = (G,x) -> ForwardDiff.gradient!(G,neg_log_backwardmess,x)
    res = optimize(neg_log_backwardmess,grad_func!, mean(q_in),LBFGS(),Optim.Options(iterations=20);inplace=true)#,Optim.Options(iterations=20))
    m_z = res.minimizer
    W_z = Zygote.hessian(neg_log_backwardmess, m_z) 
    
    return MvNormalWeightedMeanPrecision(W_z * m_z, W_z)
end