export BufferUniSGP
mutable struct BufferUniSGP{D,M} 
    qv      :: D 
    meta    :: M
end 

function ReactiveMP.prod(::GenericProd, left::NormalDistributionsFamily, right::BufferUniSGP)
    marginal_v = ReactiveMP.prod(GenericProd(),left,right.qv)
    right.meta.counter += 1
    if right.meta.counter == right.meta.N 
        μ_v, Σ_v = mean_cov(marginal_v)
        mul!(Σ_v,μ_v,μ_v',1,1) 
        Uv = fastcholesky!(Σ_v).U
        right.meta.Uv = Uv
        right.meta.counter = 0
    end
    return marginal_v
end

# rule for "v" edge (univariate case)
 
@rule UniSGP(:v, Marginalisation) (q_out::UnivariateNormalDistributionsFamily, q_in::UnivariateNormalDistributionsFamily, q_w::Any,q_θ::PointMass, meta::UniSGPMeta) = begin
    w = mean(q_w)
    μ_y = mean(q_out)
    θ = mean(q_θ)
    kernel = getKernel(meta)
    
    # Ψ1_trans = similar(meta.Ψ1_trans)
    # Ψ2 = similar(meta.Ψ2)
    Ψ1_trans = approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ),meta.Xu, [x]),q_in) 
    Ψ2 =  approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in) + 1e-8*I
    
    Ψ1_trans .*= μ_y * w #weighted-mean μ_y * w * Ψ1_transpose
    Ψ2 .*= w  #precision W_v = w * Ψ2
    return BufferUniSGP(MvNormalWeightedMeanPrecision(vec(Ψ1_trans), Ψ2),meta)
end

@rule UniSGP(:v, Marginalisation) (q_out::PointMass, q_in::PointMass, q_w::Any,q_θ::PointMass,meta::UniSGPMeta) = begin
    w = mean(q_w)
    θ = mean(q_θ)
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    kernel = getKernel(meta)

    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    mul!(meta.Ψ2,Ψ1_trans,Ψ1_trans',w,0) #W = w * Ψ1_trans * Ψ1_trans'
    Ψ1_trans .*= μ_y * w
    return BufferUniSGP(MvNormalWeightedMeanPrecision(vec(Ψ1_trans), meta.Ψ2),meta)
end

@rule UniSGP(:v, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, q_in::PointMass, q_w::Any, q_θ::PointMass, meta::UniSGPMeta) = begin 
    w = mean(q_w)
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    θ = mean(q_θ)
    kernel = getKernel(meta)

    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])
    mul!(meta.Ψ2,Ψ1_trans,Ψ1_trans',w,0) #W = w * Ψ1_trans * Ψ1_trans'
    Ψ1_trans .*= μ_y * w
    return BufferUniSGP(MvNormalWeightedMeanPrecision(vec(Ψ1_trans), meta.Ψ2),meta)
end
########