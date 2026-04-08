export GPCache
export getcache, getcache_lowermatrix
export mul_A_B!, mul_A_B_A!, mul_A_B_At!, mul_A_v!
#---- Define cache ----#

"""
    GPCache

Pre-allocated matrix/vector storage to avoid repeated heap allocations during message passing.
Holds dictionaries for matrices, vectors, and lower-triangular factors keyed by `(Symbol, size)` tuples.
Create with `GPCache()`.
"""
struct GPCache
    cache_matrices::Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}
    cache_vectors::Dict{Tuple{Symbol, Int}, Vector{Float64}}
    cache_LowerTriangular::Dict{Tuple{Symbol,Int}, LowerTriangular{Float64, Matrix{Float64}}}
end

GPCache() = GPCache(Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}(), Dict{Tuple{Symbol, Int}, Vector{Float64}}(), Dict{Tuple{Symbol,Int}, LowerTriangular{Float64, Matrix{Float64}}}())

"""
    getcache(cache::GPCache, label::Tuple{Symbol, Tuple{Int, Int}})
    getcache(cache::GPCache, label::Tuple{Symbol, Int})

Retrieve (or lazily allocate) a cached matrix or vector from `cache`, keyed by `label`.
"""
function getcache(cache::GPCache, label::Tuple{Symbol, Tuple{Int, Int}})
    return get!(() -> Matrix{Float64}(undef, label[2]), cache.cache_matrices, label)
end

function getcache(cache::GPCache, label::Tuple{Symbol, Int})
    return get!(() -> Vector{Float64}(undef, label[2]), cache.cache_vectors, label)
end

"""
    getcache_lowermatrix(cache::GPCache, label::Tuple{Symbol, Int})

Retrieve (or lazily allocate) a cached `LowerTriangular` matrix from `cache`.
"""
function getcache_lowermatrix(cache::GPCache, label::Tuple{Symbol, Int})
    return get!(() -> LowerTriangular{Float64}(rand(label[2],label[2])), cache.cache_LowerTriangular, label)
end

"""
    mul_A_B!(cache::GPCache, A, B, sizes...)

In-place matrix multiplication `A * B` using a pre-allocated buffer from `cache`.
"""
function mul_A_B!(cache::GPCache, A::AbstractArray, B, sizeA1::Int, sizeB2::Int)
    AB = getcache(cache, (:ABdiff, (sizeA1,sizeB2)))
    return mul!(AB, A, B)
end

function mul_A_B!(cache::GPCache, A::Array, B, size1::Int)
    #multiply 2 matrices with the same size
    AB = getcache(cache, (:AB, (size1, size1)))
    return mul!(AB, A, B)
end

"""
    mul_A_B_A!(cache::GPCache, A, B, size1)

In-place computation of `A * B * A` using cached buffers.
"""
function mul_A_B_A!(cache::GPCache, A::Matrix, B::Matrix, size1::Int)
    #A, B are square matrices with a same size
    AB = getcache(cache, (:AB, (size1, size1)))
    ABA = getcache(cache, (:ABA, (size1, size1)))
    mul!(AB, A, B)

    return mul!(ABA, AB, A)
end

"""
    mul_A_B_At!(cache::GPCache, A, B, sizeA1, sizeB1)

In-place computation of `A * B * A'` using cached buffers.
"""
function mul_A_B_At!(cache::GPCache, A::Matrix, B::Matrix, sizeA1::Int, sizeB1::Int)
    # A: matrix with size (sizeA1,sizeB1)
    # B: matrix with size (sizeB1,sizeB1)
    AB = getcache(cache, (:AB, (sizeA1, sizeB1)))
    ABAt = getcache(cache, (:ABA, (sizeA1, sizeA1)))
    mul!(AB, A, B)
    return mul!(ABAt, AB, A')
end
"""
    mul_A_v!(cache::GPCache, A, v, sizeA1)

In-place matrix-vector multiplication `A * v` using a cached buffer.
"""
function mul_A_v!(cache::GPCache, A::Matrix, v::Vector, sizeA1::Int)
    Av = getcache(cache, (:Av, sizeA1))
    return mul!(Av, A, v)
end