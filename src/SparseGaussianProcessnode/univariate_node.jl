# This file defines the Sparse Gaussian Process (SGP) node for univariate case
# In particular, y = f(x), where y ∈ ℝ is a scalar, and x ∈ ℝᴰ

export UniSGP, UniSGPMeta
export getmethod, getInducingInput, getKernel, getΨ0, getΨ1_trans, getΨ2, getUv, getKuuL, getcounter, getN 

struct UniSGP end 

@node UniSGP Stochastic [ out, in, v , w, θ] # out: output, in: input,  v: transformed-inducing points Kuu_inv * u , w: precision of process noise 

# Convenience type to capture acceptable types for inputs and outputs
IN_OUT = Union{Real, Array{<:Number}, PointMass, UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily}

#---- Define average energy ----#
@average_energy UniSGP (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_w::NOISE_w, q_θ::PointMass, meta::UniSGPMeta,) = begin
    w_bar = mean(q_w)
    E_logw = mean(log, q_w)
    μ_y, Σ_y = mean_cov_scalar_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans) 
    Uv = fastcholesky(Σ_v + μ_v * μ_v').U

    mf = getMeanFn(meta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = (meta.KuuL * transpose(meta.KuuL)) \ mxu

    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    α = meta.KuuL \ Ψ1_trans 
    Ψ0 .-= jdotavx(α,α)
    
    mul!(meta.Ψ2,Ψ1_trans,Ψ1_trans')

    I1 = Ψ0[1]

    I4 = (μ_y^2 
        - 2*μ_y*jdotavx(Ψ1_trans, μ_v - Ku_mxu)
        + mx^2
        + tr(Uv' * Uv * meta.Ψ2)
        + jdotavx(Ku_mxu, meta.Ψ2 * Ku_mxu)
        - 2*mx*jdotavx(Ψ1_trans, Ku_mxu)
        + 2*mx*jdotavx(Ψ1_trans, μ_v)
        - 2*jdotavx(μ_v, meta.Ψ2*Ku_mxu))
    I4 = clamp(I4, 1e-12, 1e12)
    return 0.5*(w_bar*(I1+I4) - E_logw + log(2π)) 
end