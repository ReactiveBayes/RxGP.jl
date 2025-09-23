export jdotavx, create_blockmatrix

function jdotavx(a::T, b::F) where {T<:AbstractArray, F <: AbstractArray}
    s = zero(eltype(a))
    @turbo for i ∈ eachindex(a, b)
        s += a[i] * b[i]
    end
    s
end

function create_blockmatrix(A,d,M)
    return [view(A,i:i+M-1,j:j+M-1) for i=1:M:M*d, j=1:M:M*d]
end