# rule for "w" edge (univariate case)
@rule UniSGP(:w, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_θ::PointMass, meta::UniSGPMeta,) = begin
    μ_y, Σ_y = mean_cov_scalar_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    μ_v, Σ_v = mean_cov(q_v)
    mf = getMeanFn(meta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Uv = fastcholesky(Σ_v + μ_v * μ_v').U

    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans) 
    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ), meta.Xu, [μ_in]) # col vector

    α = meta.KuuL \ Ψ1_trans
    Ψ0 .-= jdotavx(α,α) #I1

    mul!(meta.Ψ2, Ψ1_trans, Ψ1_trans') # Ψ2 = Ψ1_trans * Ψ1_trans' (matrix)
    Ku_mxu = (meta.KuuL * transpose(meta.KuuL)) \ mxu
    Ψ2_Ku_mxu = meta.Ψ2 * Ku_mxu

    Rv_Ψ1_trans = Uv * Ψ1_trans

    I4 = (
        μ_y^2 
        - 2*μ_y*( mx + jdotavx(Ψ1_trans, (μ_v - Ku_mxu)) )
        + mx^2 
        + jdotavx(Rv_Ψ1_trans, Rv_Ψ1_trans) # tr(Rv * Ψ2)
        + jdotavx(Ku_mxu, Ψ2_Ku_mxu)
        + 2*mx*jdotavx(Ψ1_trans, μ_v) 
        - 2*mx*jdotavx(Ψ1_trans, Ku_mxu) 
        - 2*jdotavx(μ_v, Ψ2_Ku_mxu)
        )

    return GammaShapeRate(1.5, 0.5*(Ψ0[1] + I4))
end