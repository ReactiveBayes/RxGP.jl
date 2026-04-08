# This file defines the Sparse Gaussian Process (SGP) node for multivariate case
# y = f(x), where y has dimension D_y, and x has dimension D_x

export MultiSGP

"""
    MultiSGP

Multivariate variational sparse Gaussian process (VSGP) factor node for
vector-valued observations ``\\mathbf{y} \\in \\mathbb{R}^D``. The multi-output
structure is built from the intrinsic coregionalization model (ICM) with
coregionalization matrix ``C_c = I_D``, so that the matrix-valued kernel is
``\\mathcal{K}(\\mathbf{x}, \\mathbf{x}') = C_c \\otimes k_{\\boldsymbol{\\theta}}(\\mathbf{x}, \\mathbf{x}')``.

The composite node function collapses the GP prior into the observation model:

```math
f_{\\mathrm{SGP}}(\\mathbf{y}, \\mathbf{x}, \\mathbf{v}, W, \\boldsymbol{\\theta})
= \\exp\\!\\Bigl(-\\tfrac{1}{2}\\operatorname{tr}(W\\, A_{\\mathbf{x},\\boldsymbol{\\theta}})\\Bigr)\\;
  \\mathcal{N}\\!\\bigl(\\mathbf{y} \\mid B_{\\mathbf{x},\\boldsymbol{\\theta}}\\,\\mathbf{v},\\; W^{-1}\\bigr)
```

where

```math
B_{\\mathbf{x},\\boldsymbol{\\theta}} = C_c \\otimes k_{\\boldsymbol{\\theta}}(\\mathbf{x}, X_u)
  \\in \\mathbb{R}^{D \\times DM},
```
```math
A_{\\mathbf{x},\\boldsymbol{\\theta}}
  = C_c \\otimes k_{\\boldsymbol{\\theta}}(\\mathbf{x}, \\mathbf{x})
    - B_{\\mathbf{x},\\boldsymbol{\\theta}}\\,
      K_u^{-1}\\,
      B_{\\mathbf{x},\\boldsymbol{\\theta}}^\\top
  \\in \\mathbb{R}^{D \\times D},
```

with ``K_u = C_c \\otimes k_{\\boldsymbol{\\theta}}(X_u, X_u)`` and
``\\mathbf{v} = K_u^{-1}\\mathbf{u} \\in \\mathbb{R}^{DM}``.
The computational complexity is ``\\mathcal{O}(DNM^2)``.

**Edges:** `[out, in, v, w, θ]` — output, input, stacked transformed inducing variables, Wishart noise precision, kernel hyperparameters.
"""
struct MultiSGP end 

@node MultiSGP Stochastic [ out, in, v , w, θ] ## out: output, in: input,  v: transformed-inducing points Kuu_inv * u , w: precision of process noise 

#---- Average energy ----#
@average_energy MultiSGP (q_out::MultivariateNormalDistributionsFamily, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W_bar = mean(q_w) 
    θ = mean(q_θ)
    E_logW = mean(logdet,q_w)
    Xu = getInducingInput(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    Kuu_inverse = getKuuInverse(meta)
    method = getmethod(meta)
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W_bar,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W_bar)
    Ry = Σ_y + μ_y * μ_y'

    Ψ0 = getΨ0(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 
    approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)[1]
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in)

    return  0.5*D*log(2π) - 0.5*E_logW + 0.5*tr(W_bar*Ry)+ 0.5 * tr(W_bar) * (Ψ0[1] - sum(Kuu_inverse .* Ψ2)) - sum(sumdiagV .* Ψ1_trans) + 0.5 * sum(Ψ2 .* sumRvblk_W)
end


@average_energy MultiSGP (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W_bar = mean(q_w) 
    θ = mean(q_θ)
    E_logW = mean(logdet,q_w)
    Xu = getInducingInput(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    Kuu_inverse = getKuuInverse(meta)
    method = getmethod(meta)
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W_bar,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W_bar)
    Ry = Σ_y + μ_y * μ_y'

    Ψ0 = getΨ0(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 

    kernelmatrix!(Ψ0,kernel(θ), [mean(q_in)], [mean(q_in)])
    kernelmatrix!(Ψ1_trans,kernel(θ), Xu,[mean(q_in)])
    Ψ2 = kernelmatrix(kernel(θ), Xu, [mean(q_in)]) * kernelmatrix(kernel(θ), [mean(q_in)], Xu)

    return  0.5*D*log(2π) - 0.5*E_logW + 0.5*tr(W_bar*Ry)+ 0.5 * tr(W_bar) * (Ψ0[1] - sum(Kuu_inverse .* Ψ2)) - sum(sumdiagV .* Ψ1_trans) + 0.5 * sum(Ψ2 .* sumRvblk_W)
end


@average_energy MultiSGP (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    θ = mean(q_θ)
    W_bar = mean(q_w) 
    E_logW = log(det(W_bar))
    Xu = getInducingInput(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    Kuu_inverse = getKuuInverse(meta)
    method = getmethod(meta)
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W_bar,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W_bar)
    Ry = μ_y * μ_y'

    Ψ0 = getΨ0(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 

    approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)[1]
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in)

    return  0.5*D*log(2π) - 0.5*E_logW + 0.5*tr(W_bar*Ry)+ 0.5 * tr(W_bar) * (Ψ0[1] - sum(Kuu_inverse .* Ψ2)) - sum(sumdiagV .* Ψ1_trans) + 0.5 * sum(Ψ2 .* sumRvblk_W)
end

@average_energy MultiSGP (q_out::PointMass, q_in::MultivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::Wishart,q_θ::PointMass, meta::MultiSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    θ = mean(q_θ)
    W_bar = mean(q_w) 
    E_logW = mean(logdet,q_w)
    Xu = getInducingInput(meta)
    cache = getGPCache(meta)
    kernel = getKernel(meta)
    D = length(μ_y)
    M = length(Xu) #number of inducing points
    Kuu_inverse = getKuuInverse(meta)
    method = getmethod(meta)
    Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D) #Rv = Σ_v + μ_v * μ_v'
    V = mul_A_B!(cache,μ_v,μ_y',M*D,D) |> (x) -> mul_A_B!(GPCache(),x, W_bar,M*D,D)
    sumdiagV = sum_diagonal_M(V,M)
    sumRvblk_W = sum(create_blockmatrix(Σ_v,D,M) .* W_bar)
    Ry = μ_y * μ_y'

    Ψ0 = getΨ0(meta)
    Ψ1_trans = getΨ1_trans(meta) 
    Ψ2 = getΨ2(meta) 

    approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)[1]
    approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]),q_in) 
    approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in)

    return  0.5*D*log(2π) - 0.5*E_logW + 0.5*tr(W_bar*Ry)+ 0.5 * tr(W_bar) * (Ψ0[1] - sum(Kuu_inverse .* Ψ2)) - sum(sumdiagV .* Ψ1_trans) + 0.5 * sum(Ψ2 .* sumRvblk_W)
end