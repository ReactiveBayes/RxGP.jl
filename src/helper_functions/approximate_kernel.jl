export approximate_kernel_expectation

function approximate_kernel_expectation(method::GenUnscented, g::Function, q::D) where {D <: NormalDistributionsFamily}
    return approximate_expectation(method, q, g)
end