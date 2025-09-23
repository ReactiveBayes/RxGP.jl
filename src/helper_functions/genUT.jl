import LoopVectorization: vmapreduce
import SparseArrays: blockdiag,sparse, spdiagm
import ExponentialFamily: mean_cov

export GenUT, GenUnscented
"""
The `GenUnscented` structure defines the approximation method of Generalized Unscented Transform. 
"""
struct GenUnscented <: AbstractApproximationMethod end


"""An alias for the [`GenUnscented`](@ref) approximation method."""
const GenUT = GenUnscented

"""An alias for the [`GenUnscented`](@ref) approximation method."""
const GenUnscentedTransform = GenUnscented

# add univariate case
function approximate_expectation(method::GenUnscented, q::D, f::F) where {F, D <: UnivariateDistribution}
    m = mean(q)
    V = var(q)
    S = skewness(q)
    K = kurtosis(q, false)
    (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)
    return sum(weights .* f.(sigma_points))
end

function approximate_expectation(method::GenUnscented, q, f::F) where {F}
    m = mean(q)
    V = cov(q)
    S = skewness(q)
    K = kurtosis(q, false)
    (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)
    return sum(weights .* f.(sigma_points))
end

function approximate_expectation(method::GenUnscented, q::Tuple, f::F) where {F}
    ms = mean.(q)
    ds = size.(ms)
    m = vcat(ms...)
    V = vmapreduce(d -> typeof(d) <: Matrix ? sparse(d) : spdiagm([d]),blockdiag ,cov.(q))
    S = vcat(skewness.(q)...)
    K = vcat(kurtosis.(q, false)...)
    (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)
    f_sigma = [f(__splitjoin(sp, ds)...) for sp in sigma_points] #
    return sum(weights .* f_sigma)
end

function Unscented_approximate(method::GenUnscented, g::G, means::Tuple, covs::Tuple,skewnesses::Tuple, kurtosises::Tuple) where {G}
    (m, V, S, K, _) = unscented_statistics(method, Val(false), g, means, covs, skewnesses, kurtosises)
    return (m, V, S, K)
end

function unscented_statistics(method::GenUnscented, g::G, means::Tuple, covs::Tuple,skewnesses::Tuple, kurtosises::Tuple) where {G}
    # By default we compute the `C` component, thus `Val(true)`
    return unscented_statistics(method, Val(true), g, means, covs, skewnesses, kurtosises)
end

# Single univariate variable
function unscented_statistics(method::GenUnscented, ::Val{C}, g::G, means::Tuple{Real}, covs::Tuple{Real},skewnesses::Tuple{Real}, kurtosises::Tuple{Real}) where {C, G}
    m = first(means)
    V = first(covs)
    S = first(skewnesses)
    K = first(kurtosises)

    (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)

    g_sigma = g.(sigma_points)
    m_tilde = sum(weights .* g_sigma)
    V_tilde = sum(weights .* (g_sigma .- m_tilde) .^ 2)
    S_tilde = sum(weights .* (g_sigma .- m_tilde) .^ 3)
    K_tilde = sum(weights .* (g_sigma .- m_tilde) .^ 4)

    # Compute `C_tilde` only if `C === true`
    C_tilde = C ? sum(weights .* (sigma_points .- m) .* (g_sigma .- m_tilde)) : nothing

    return (m_tilde, V_tilde, S_tilde, K_tilde, C_tilde)
end

# Single multivariate variable
function unscented_statistics(method::GenUnscented, ::Val{C}, g::G, means::Tuple{AbstractVector}, covs::Tuple{AbstractMatrix},skewnesses::Tuple{AbstractVector}, kurtosises::Tuple{AbstractVector} ) where {C, G}
    m = first(means)
    V = first(covs)
    S = first(skewnesses)
    K = first(kurtosises)

    (sigma_points, weights) = sigma_points_weights(method, m, V, S, K)

    d = length(m)

    g_sigma = g.(sigma_points)
    @inbounds m_tilde = sum(weights[k + 1] * g_sigma[k + 1] for k in 0:(2d))
    @inbounds V_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d))
    @inbounds S_tilde = sum(weights[k + 1]*(g_sigma[k + 1] - m_tilde).^3 for k in 0:2d)
    @inbounds K_tilde = sum(weights[k + 1]*(g_sigma[k + 1] - m_tilde).^4 for k in 0:2d)
    # Compute `C_tilde` only if `C === true`
    @inbounds C_tilde = C ? sum(weights[k + 1] * (sigma_points[k + 1] - m) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) : nothing

    return (m_tilde, V_tilde, S_tilde, K_tilde, C_tilde)
end

# Multiple inbounds of possibly mixed variate type
function unscented_statistics(method::GenUnscented, ::Val{C}, g::G, ms::Tuple, Vs::Tuple,Ss::Tuple, Ks::Tuple) where {C, G}
    ds     = size.(ms)
    m      = vcat(ms...)
    S      = vcat(Ss...)
    K      = vcat(Ks...)
    V      = vmapreduce(d -> typeof(d) <: Matrix ? sparse(d) : spdiagm([d]),blockdiag ,Vs)
    (sigma_points, weights ) = sigma_points_weights(method, m, V, S, K)
    g_sigma = [g(__splitjoin(sp, ds)...) for sp in sigma_points] # Unpack each sigma point in g

    d = sum(prod.(ds)) # Dimensionality of joint
    @inbounds m_tilde = sum(weights[k + 1] * g_sigma[k + 1] for k in 0:(2d)) # Vector
    @inbounds V_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) # Matrix
    @inbounds S_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde).^3 for k in 0:(2d))
    @inbounds K_tilde = sum(weights[k + 1] * (g_sigma[k + 1] - m_tilde).^4 for k in 0:(2d))
    # # Compute `C_tilde` only if `C === true`
    @inbounds C_tilde = C ? sum(weights[k + 1] * (sigma_points[k + 1] - m) * (g_sigma[k + 1] - m_tilde)' for k in 0:(2d)) : nothing

    return (m_tilde, V_tilde, S_tilde, K_tilde, C_tilde)
end


#Univariate sigma points and weights
function sigma_points_weights(::GenUnscented, m::Real, V::Real,S::Real,K::Real)
    L     = sqrt(V)
    invL3 = inv(L^3) 
    u     = (1/2)*(-S*invL3 + (1/V)*sqrt(4*K - 3*(S^2)/V))
    v     = u + S*invL3
    auxiliary    = inv(v*(u+v))
    sigma_points = (m, m - u*L, m + v*L)
    weights      = (1-auxiliary*(v/u +1)  , (v/u)*auxiliary, auxiliary)

    return (sigma_points, weights)
end

#Multivariate sigma points and weights
function sigma_points_weights(::GenUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector,K::AbstractVector)
    d     = length(m)
    T     = promote_type(eltype(m), eltype(V),eltype(S),eltype(K))
    L     = cholsqrt(V)
    L3    = L.^3
    invL3 = cholinv(L3)
    invL4 = cholinv(L3 .* L3)
    determinant = 4*invL4*K - 3*(invL3*S).^2
    u     = (1/2)*(-invL3*S + sqrt.(determinant ) )
    v     = u + invL3*S
    sigma_points = Vector{Vector{T}}(undef, 2 * d + 1)
    weights      = Vector{T}(undef, 2 * d + 1)
    sigma_points[1] = m
    @inbounds for i in 1:d
        @views sigma_points[i+1]  = m - L[:, i]*u[i]
        @views sigma_points[i+d+1] = m + L[:, i]*v[i]
    end
    @inbounds weights[d+2:end] .=  1 ./ v ./ ( u + v) 
    @inbounds weights[2:d+1]   .= weights[d+2:end] .* (v ./ u)
    weights[1] = 1 - sum(weights[2:end])

    return (sigma_points, weights)
end