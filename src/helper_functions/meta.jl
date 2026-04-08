export UniSGPMeta, MultiSGPMeta
export getmethod, getMeanFn, getInducingInput, getΨx, getΨxx, getΨ0, getΨ1_trans, getΨ2, getΨ3, getLm_fn, getKxx_fn, getKxu_fn, getKuuF, getKernel, get_dims_data, get_dims_theta, getUv, getcounter, getN
export getKuuInverse, getGPCache

#---- Define metas -----# 
## create UniSGP meta  
# Lm_fn(x)         — L*m(x):       operated mean,         returns P-vector
# Kxx_fn(x,θ)      — L1*L2*k(x,x): operated auto-kernel,  returns P×P matrix
# Kxu_fn(x,θ,Xu)   — L1*k(x,Xu):   operated cross-kernel, returns P×M matrix
# P is determined by the linear operator L (e.g. P=1 for identity, P=D for gradient,
# P=1+D for stacked identity+gradient). UniSGP (scalar node) leaves these as nothing.
mutable struct UniSGPMeta{I,E,D,C,CF,K}
    method          :: Union{Nothing,AbstractApproximationMethod}
    mean_fn         :: Function
    Xu              :: I
    Ψx              :: Float64
    Ψxx             :: Float64
    Ψ0              :: Float64
    Ψ1_trans        :: Matrix{Float64}
    Ψ2              :: Matrix{Float64}
    Ψ3              :: Matrix{Float64}
    Lm_fn           :: E        # L*m(x):       operated mean function
    Kxx_fn          :: D        # L1*L2*k(x,x): operated auto-kernel
    Kxu_fn          :: C        # L1*k(x,Xu):   operated cross-kernel
    KuuF            :: CF
    kernel          :: K
    dims_data       :: Int      # P: output dimension under the linear operator
    dims_theta      :: Int
    Uv              :: Matrix{Float64}
    counter         :: Int
    N               :: Int
end
getmethod(meta::UniSGPMeta) = meta.method
getMeanFn(meta::UniSGPMeta) = meta.mean_fn
getInducingInput(meta::UniSGPMeta) = meta.Xu
getΨx(meta::UniSGPMeta) = meta.Ψx
getΨxx(meta::UniSGPMeta) = meta.Ψxx
getΨ0(meta::UniSGPMeta) = meta.Ψ0
getΨ1_trans(meta::UniSGPMeta) = meta.Ψ1_trans
getΨ2(meta::UniSGPMeta) = meta.Ψ2
getΨ3(meta::UniSGPMeta) = meta.Ψ3
getLm_fn(meta::UniSGPMeta) = meta.Lm_fn
getKxx_fn(meta::UniSGPMeta) = meta.Kxx_fn
getKxu_fn(meta::UniSGPMeta) = meta.Kxu_fn
getKuuF(meta::UniSGPMeta) = meta.KuuF
getKernel(meta::UniSGPMeta) = meta.kernel
get_dims_data(meta::UniSGPMeta) = meta.dims_data
get_dims_theta(meta::UniSGPMeta) = meta.dims_theta
getUv(meta::UniSGPMeta) = meta.Uv
getcounter(meta::UniSGPMeta) = meta.counter
getN(meta::UniSGPMeta) = meta.N

## create MultiSGP meta  
mutable struct MultiSGPMeta{I,K}
    method      :: Union{Nothing,AbstractApproximationMethod}
    Xu          :: I    # inducing inputs
    Ψ0          :: Matrix{Float64}
    Ψ1_trans    :: Matrix{Float64}
    Ψ2          :: Matrix{Float64}
    Kuu_inverse :: Matrix{Float64}
    kernel      :: K
    GPCache     :: Union{Nothing,GPCache}
end
getInducingInput(meta::MultiSGPMeta) = meta.Xu
# getCoregionalizationMatrix(meta::MultiSGPMeta) = meta.C
getΨ0(meta::MultiSGPMeta) = meta.Ψ0 
getΨ1_trans(meta::MultiSGPMeta) = meta.Ψ1_trans
getΨ2(meta::MultiSGPMeta) = meta.Ψ2
getKuuInverse(meta::MultiSGPMeta) = meta.Kuu_inverse
getKernel(meta::MultiSGPMeta) = meta.kernel
getGPCache(meta::MultiSGPMeta) = meta.GPCache
getmethod(meta::MultiSGPMeta) = meta.method