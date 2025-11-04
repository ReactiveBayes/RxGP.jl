# This file defines the Sparse Gaussian Process (SGP) node for univariate case
# In particular, y = f(x), where y ∈ ℝ is a scalar, and x ∈ ℝᴰ

export UniSGP, UniSGPMeta
export getmethod, getInducingInput, getKernel, getΨ0, getΨ1_trans, getΨ2, getUv, getKuuL, getcounter, getN 

struct UniSGP end 

@node UniSGP Stochastic [ out, in, v , w, θ] # out: output, in: input,  v: transformed-inducing points Kuu_inv * u , w: precision of process noise 

#---- Define meta -----# 
## create UniSGP meta  
mutable struct UniSGPMeta{I,K}
    method      :: Union{Nothing,AbstractApproximationMethod}
    Xu          :: I    # inducing inputs
    Ψ0          :: Matrix{Float64}
    Ψ1_trans    :: Matrix{Float64}
    Ψ2          :: Matrix{Float64}
    KuuL        :: AbstractArray
    kernel      :: K
    Uv          :: AbstractArray
    counter     :: Int
    N           :: Int
end
getmethod(meta::UniSGPMeta) = meta.method
getInducingInput(meta::UniSGPMeta) = meta.Xu
getKernel(meta::UniSGPMeta) = meta.kernel
getΨ0(meta::UniSGPMeta) = meta.Ψ0
getΨ1_trans(meta::UniSGPMeta) = meta.Ψ1_trans
getΨ2(meta::UniSGPMeta) = meta.Ψ2
getUv(meta::UniSGPMeta) = meta.Uv # Cholesky upper triangular of Rv = μ_v * μ_v' + Σ_v
getKuuL(meta::UniSGPMeta) = meta.KuuL
getcounter(meta::UniSGPMeta) = meta.counter
getN(meta::UniSGPMeta) = meta.N #number of observations


#---- Define average energy ----# TEST
#### general case 
@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate,q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y, v_y = mean_var(q_out)
    μ_v = mean(q_v)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans)
    Ψ2 = similar(meta.Ψ2)
    # approximate_kernel_expectation!(Ψ0,getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)
    # approximate_kernel_expectation!(Ψ1_trans,getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]),q_in)
    # approximate_kernel_expectation!(Ψ2,getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I 
    
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)
    Ψ1_trans = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-8*I
    I1 = clamp(Ψ0[1] - tr(meta.KuuL' \ (meta.KuuL \ Ψ2)),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * dot(Ψ1_trans, μ_v) + tr(meta.Uv' * meta.Uv * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

# precision w is a Pointmass
@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass, q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = log(w_bar)
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Kuu_inverse = inv(kernelmatrix(kernel(θ),Xu) .+ 1e-8)
    
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in) .+ 1e-8
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) .+ 1e-8 
    
    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

#### regression problem
@average_energy UniSGP (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate, q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y = mean(q_out)
    μ_v = mean(q_v)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans) 

    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    α = meta.KuuL \ Ψ1_trans 
    Ψ0 .-= jdotavx(α,α)

    I2 = μ_y^2 - 2*μ_y*jdotavx(Ψ1_trans,μ_v) 
    mul!(Ψ1_trans,meta.Uv,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 
    return 0.5*(Ψ0[1]*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

# precision w is a PointMass
@average_energy UniSGP (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass, q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = log(w_bar)
    μ_y = mean(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)

    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])


    α = meta.KuuL \ Ψ1_trans
    Ψ0 .-= jdotavx(α,α)

    mul!(Σ_v,μ_v,μ_v',1,1)  # Σ_v = Σ_v + μ_v * μ_v'
    Lu = cholesky!(Σ_v).U
    I2 = μ_y^2 - 2*μ_y*jdotavx(Ψ1_trans,μ_v)
    mul!(Ψ1_trans,Lu,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 
    return 0.5*(Ψ0[1]*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

# classification problem
@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::GammaShapeRate, q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_y, v_y = mean_var(q_out)

    μ_v = mean(q_v)
    μ_in = mean(q_in)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans)

    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])


    α = meta.KuuL \ Ψ1_trans
    Ψ0 .-= jdotavx(α,α)
    
    I2 = μ_y^2 + v_y - 2*μ_y*jdotavx(Ψ1_trans,μ_v)
    mul!(Ψ1_trans,meta.Uv,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 
    return 0.5*(Ψ0[1]*w_bar - E_logw + log(2π) + I2 * w_bar) 
end

@average_energy UniSGP (q_out::UnivariateNormalDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily, q_w::PointMass, q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = log(w_bar)
    μ_y, v_y = mean_var(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    μ_in = mean(q_in)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    Kuu_inverse = inv(kernelmatrix(kernel(θ),Xu) .+ 1e-8)
    
    Ψ0 = getindex(kernelmatrix(kernel(θ), [μ_in], [μ_in]),1)
    Ψ1 = kernelmatrix(kernel(θ), [μ_in], Xu) .+ 1e-8
    Ψ2 = kernelmatrix(kernel(θ), Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], Xu) .+ 1e-8

    I1 = clamp(Ψ0 - tr(Kuu_inverse * Ψ2),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * (Ψ1 * μ_v)[1] + tr((Σ_v + μ_v * μ_v') * Ψ2 ), 1e-12,1e12)

    return 0.5*(I1*w_bar - E_logw + log(2π) + I2 * w_bar) 
end
