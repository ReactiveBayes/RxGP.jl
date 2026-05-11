export UniSGPMeta, MultiSGPMeta
export getmethod, getMeanFn, getInducingInput, getΨx, getΨxx, getΨ0, getΨ1_trans, getΨ2, getΨ3, getLm_fn, getKxx_fn, getKxu_fn, getKuuF, getKernel, get_dims_input, get_dims_output, get_dims_theta, getUv, getcounter, getN
export getKuuInverse, getGPCache

#---- Define metas -----# 

"""
    UniSGPMeta

Metadata for the [`UniSGP`](@ref) and [`UniSGP_dID`](@ref) factor nodes. Stores the approximation method, kernel configuration,
inducing point locations, precomputed kernel matrices, operator functions, and mutable workspace for kernel expectations.

Construct via [`get_UniSGPMeta`](@ref) rather than calling the constructor directly.
"""
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
    dims_input      :: Int      # D: input dimension
    dims_output     :: Int      # P: output dimension under the linear operator
    dims_theta      :: Int
    Uv              :: Matrix{Float64}
    counter         :: Int
    N               :: Int
end

"""Return the approximation method."""
getmethod(meta::UniSGPMeta) = meta.method
"""Return the prior mean function."""
getMeanFn(meta::UniSGPMeta) = meta.mean_fn
"""Return the inducing input locations ``\\mathbf{X}_u``."""
getInducingInput(meta::UniSGPMeta) = meta.Xu
"""Return the cached scalar kernel expectation ``\\Psi_x``."""
getΨx(meta::UniSGPMeta) = meta.Ψx
"""Return the cached scalar kernel expectation ``\\Psi_{xx}``."""
getΨxx(meta::UniSGPMeta) = meta.Ψxx
"""Return the cached kernel diagonal expectation ``\\Psi_0``."""
getΨ0(meta::UniSGPMeta) = meta.Ψ0
"""Return the cached transposed cross-kernel expectation ``\\Psi_1^\\top``."""
getΨ1_trans(meta::UniSGPMeta) = meta.Ψ1_trans
"""Return the cached quadratic kernel expectation ``\\Psi_2``."""
getΨ2(meta::UniSGPMeta) = meta.Ψ2
"""Return the cached kernel expectation ``\\Psi_3``."""
getΨ3(meta::UniSGPMeta) = meta.Ψ3
"""Return the operated mean function ``\\tilde{L}m(x)``."""
getLm_fn(meta::UniSGPMeta) = meta.Lm_fn
"""Return the operated auto-kernel function ``\\tilde{K}_{xx}(x,\\theta)``."""
getKxx_fn(meta::UniSGPMeta) = meta.Kxx_fn
"""Return the operated cross-kernel function ``\\tilde{K}_{xu}(x,\\theta,X_u)``."""
getKxu_fn(meta::UniSGPMeta) = meta.Kxu_fn
"""Return the Cholesky factorisation of ``K_{uu}``."""
getKuuF(meta::UniSGPMeta) = meta.KuuF
"""Return the kernel constructor ``\\theta \\mapsto k``."""
getKernel(meta::UniSGPMeta) = meta.kernel
"""Return the output dimension ``P`` under the linear operator."""
get_dims_output(meta::UniSGPMeta) = meta.dims_output
"""Return the input dimension ``D``."""
get_dims_input(meta::UniSGPMeta) = meta.dims_input
"""Return the number of kernel hyperparameters."""
get_dims_theta(meta::UniSGPMeta) = meta.dims_theta
"""Return the Cholesky factor of the second moment of ``\\mathbf{v}``."""
getUv(meta::UniSGPMeta) = meta.Uv
"""Return the current observation counter (for `BufferUniSGP` accumulation)."""
getcounter(meta::UniSGPMeta) = meta.counter
"""Return the total number of observations ``N``."""
getN(meta::UniSGPMeta) = meta.N

"""
    MultiSGPMeta

Metadata for the [`MultiSGP`](@ref) factor node. Stores the approximation method, kernel, inducing points,
precomputed ``K_{uu}^{-1}``, kernel expectation workspace, and a [`GPCache`](@ref) for in-place operations.
"""
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
"""Return the inducing input locations ``\\mathbf{X}_u``."""
getInducingInput(meta::MultiSGPMeta) = meta.Xu
"""Return the cached kernel expectation ``\\Psi_0``."""
getΨ0(meta::MultiSGPMeta) = meta.Ψ0 
"""Return the cached transposed cross-kernel expectation ``\\Psi_1^\\top``."""
getΨ1_trans(meta::MultiSGPMeta) = meta.Ψ1_trans
"""Return the cached quadratic kernel expectation ``\\Psi_2``."""
getΨ2(meta::MultiSGPMeta) = meta.Ψ2
"""Return the precomputed ``K_{uu}^{-1}`` matrix."""
getKuuInverse(meta::MultiSGPMeta) = meta.Kuu_inverse
"""Return the kernel constructor ``\\theta \\mapsto k``."""
getKernel(meta::MultiSGPMeta) = meta.kernel
"""Return the [`GPCache`](@ref) used for in-place matrix operations."""
getGPCache(meta::MultiSGPMeta) = meta.GPCache
"""Return the approximation method."""
getmethod(meta::MultiSGPMeta) = meta.method