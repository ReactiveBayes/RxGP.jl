# rule for "w" edge (univariate case)

@rule UniSGP(:w, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::UnivariateGaussianDistributionsFamily,
            q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass,meta::UniSGPMeta,) = begin
    μ_y, v_y = mean_var(q_out)
    μ_v = mean(q_v)
    Xu = getInducingInput(meta)
    kernel = getKernel(meta)
    θ = mean(q_θ)
    
    Ψ0 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
    Ψ1 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], meta.Xu),q_in)
    Ψ2 = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in) + 1e-8*I 

    I1 = clamp(Ψ0 - tr(meta.KuuL' \ (meta.KuuL \ Ψ2)),1e-12,1e12)
    I2 = clamp(μ_y^2 + v_y -2*μ_y * dot(Ψ1, μ_v) + tr(meta.Uv' * meta.Uv * Ψ2 ),1e-12,1e12)
    return GammaShapeRate(1.5, 0.5*(I1 + I2))
end


@rule UniSGP(:w, Marginalisation) (q_out::PointMass, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::UniSGPMeta,) = begin
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    μ_v = mean(q_v)

    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans) 
    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    α = meta.KuuL \ Ψ1_trans
    Ψ0 .-= jdotavx(α,α) #I1

    I2 = μ_y^2 - 2*μ_y*jdotavx(Ψ1_trans,μ_v)
    mul!(Ψ1_trans,meta.Uv,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 

    return GammaShapeRate(1.5, 0.5*(Ψ0[1] + I2))
end


@rule UniSGP(:w, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_v::MultivariateNormalDistributionsFamily,q_θ::PointMass, meta::UniSGPMeta,) = begin
    μ_y, v_y = mean_var(q_out)
    μ_v = mean(q_v)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)

    Ψ0 = similar(meta.Ψ0)
    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(Ψ0,kernel(θ), [μ_in], [μ_in])
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    α = meta.KuuL \ Ψ1_trans
    Ψ0 .-= jdotavx(α,α) #I1

    I2 = μ_y^2 + v_y - 2*μ_y*jdotavx(Ψ1_trans,μ_v) 
    mul!(Ψ1_trans,meta.Uv,Ψ1_trans)
    I2 += jdotavx(Ψ1_trans,Ψ1_trans) 
    return GammaShapeRate(1.5, 0.5*(Ψ0[1] + I2))
end