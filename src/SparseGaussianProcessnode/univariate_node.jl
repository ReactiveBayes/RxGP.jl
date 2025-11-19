# This file defines the Sparse Gaussian Process (SGP) node for univariate case
# In particular, y = f(x), where y ∈ ℝ is a scalar, and x ∈ ℝᴰ

export UniSGP, UniSGPMeta

struct UniSGP end 

@node UniSGP Stochastic [ out, in, v , w, θ] # out: output, in: input,  v: transformed-inducing points Kuu_inv * u , w: precision of process noise 

# Convenience type to capture acceptable types for inputs and outputs
IN_OUT = Union{Real, Array{<:Number}, PointMass, UnivariateNormalDistributionsFamily, MultivariateNormalDistributionsFamily}

#---- Define average energy ----#
@average_energy UniSGP (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_w::NOISE_w, q_θ::PointMass, meta::UniSGPMeta,) = begin
    μ_y, Σ_y = mean_cov_scalar_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    μ_v, Σ_v = mean_cov_vector_matrix(q_v)
    mf = getMeanFn(meta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Rv = Σ_v + μ_v * μ_v'
    # Uv = fastcholesky(Rv).U

    if q_in isa Distribution
        meta.Ψ0 = approximate_kernel_expectation(meta.method,(x) -> kernelmatrix(kernel(θ), [x], [x]), q_in)[1]
        meta.Ψ1_trans = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_in)
        meta.Ψ2 = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in) + 1e-8*I
    else
        meta.Ψ0 = kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
        meta.Ψ1_trans = kernelmatrix(kernel(θ), meta.Xu, [μ_in])
        meta.Ψ2 = kernelmatrix(kernel(θ), meta.Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], meta.Xu) + 1e-8*I
    end

    I1 = meta.Ψ0
    α = meta.KuuF.L \ meta.Ψ1_trans
    I1 -= jdotavx(α,α) #I1 = meta.Ψ0 - tr( meta.KuuF \ meta.Ψ2 )

    Ku_mxu = meta.KuuF \ mxu
    Ψ2_Ku_mxu = meta.Ψ2 * Ku_mxu

    # Uv_Ψ1_trans = Uv * meta.Ψ1_trans # tr(Rv * Ψ2)

    I4 = (
        μ_y^2 
        + Σ_y[1,1]
        - 2*μ_y*( mx + jdotavx(meta.Ψ1_trans, (μ_v - Ku_mxu)) )
        + mx^2 
        + tr(Rv * meta.Ψ2) # + jdotavx(Uv_Ψ1_trans, Uv_Ψ1_trans)
        + jdotavx(Ku_mxu, Ψ2_Ku_mxu)
        + 2*mx*jdotavx(meta.Ψ1_trans, μ_v) 
        - 2*mx*jdotavx(meta.Ψ1_trans, Ku_mxu) 
        - 2*jdotavx(μ_v, Ψ2_Ku_mxu)
    )
    I4 = clamp(I4, 1e-12, 1e12)

    return 0.5*(w_bar*(I1+I4) - E_logw + log(2π)) 
end