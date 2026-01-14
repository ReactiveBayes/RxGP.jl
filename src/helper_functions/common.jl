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