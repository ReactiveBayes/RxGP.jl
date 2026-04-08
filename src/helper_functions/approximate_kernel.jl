export approximate_kernel_expectation, approximate_kernel_expectation!

"""
    approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, distribution)

Approximate the expectation ``\\mathbb{E}_{q(x)}[g(x)]`` where ``x \\sim`` `distribution` using the specified approximation `method`.
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

"""
    approximate_kernel_expectation!(gbar, method::AbstractApproximationMethod, g::Function, distribution)

In-place version of [`approximate_kernel_expectation`](@ref). Writes the result into the pre-allocated array `gbar`.
"""
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