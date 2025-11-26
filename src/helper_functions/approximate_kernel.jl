export approximate_kernel_expectation!, approximate_kernel_expectation

#univariate case
function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, m::Real, P::Real)
    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)
    gbar = g(m) - g(m)
    foreach(zip(weights, points)) do (weight, point)
        gbar += weight * g(point)
    end
    return gbar
end

function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, q::D) where {D <: UnivariateDistribution}
    return approximate_kernel_expectation(method, g, mean(q), var(q))
end

function approximate_kernel_expectation(method::GenUnscented, g::Function, q::D) where {D <: UnivariateDistribution}
    return approximate_expectation(method, q, g)
end

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, m::Real, P::Real) where {K <: Array}
    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)
    gbar .= 0
    foreach(zip(weights, points)) do (weight, point)
        gbar .+= weight * g(point)
    end
    return gbar
end

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, distribution::D) where {K <: Array, D <: UnivariateDistribution}
    return approximate_kernel_expectation!(gbar, method, g, mean(distribution), var(distribution))
end


#multivariate case 

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, m::AbstractVector{T}, P::AbstractMatrix{T}) where {K <: Array, T <: Real}
    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)

    gbar .= 0
    foreach(zip(weights, points)) do (weight, point)
        axpy!(weight, g(point), gbar) # gbar = gbar + weight * g(point)
    end
    return gbar
end

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, distribution::D) where {K <: Array, D <: MultivariateDistribution}
    return approximate_kernel_expectation!(gbar, method, g, mean(distribution), cov(distribution))
end

function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, m::AbstractVector{T}, P::AbstractMatrix{T}) where {T <: Real}
    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)

    gbar = g(m) .* 0.0
    foreach(zip(weights, points)) do (weight, point)
        axpy!(weight, g(point), gbar) # gbar = gbar + weight * g(point)
    end
    return gbar
end

function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, distribution::D) where {D <: MultivariateDistribution}
    return approximate_kernel_expectation(method, g, mean(distribution), cov(distribution))
end