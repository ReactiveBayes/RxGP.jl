"""
   approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, distribution::D) where {D <: NormalDistributionsFamily}
   
Approximate the expectation E[g(x)] where x ~ distribution using the specified approximation method.
"""
function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, distribution::D) where {D <: NormalDistributionsFamily}
    return approximate_kernel_expectation(method, g, mean(distribution), cov(distribution))
end

function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, m::Union{T, AbstractVector{T}}, P::Union{T, AbstractMatrix{T}}) where {T <: Real}
    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)
    gbar = zero(g(m))
    foreach(zip(weights, points)) do (weight, point)
        gbar = gbar .+ weight .* g(point)
    end
    return gbar
end

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, distribution::D) where {K <: Array, D <: NormalDistributionsFamily}
    return approximate_kernel_expectation!(gbar, method, g, mean(distribution), cov(distribution))
end

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, m::Union{T, AbstractVector{T}}, P::Union{T, AbstractMatrix{T}}) where {K <: Array, T <: Real}
    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)

    gbar .= 0
    foreach(zip(weights, points)) do (weight, point)
        axpy!(weight, g(point), gbar) # gbar = gbar + weight * g(point)
    end
    return gbar
end

function approximate_kernel_expectation(method::GenUnscented, g::Function, q::D) where {D <: NormalDistributionsFamily}
    return approximate_expectation(method, q, g)
end