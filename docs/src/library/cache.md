```@meta
CurrentModule = RxGP
```

# [Cache utilities](@id cache-reference)

The [`GPCache`](@ref) system provides pre-allocated matrix and vector storage to avoid repeated heap allocations during message passing. This is critical for performance in large models with many observations.

## GPCache

```@docs
GPCache
```

A `GPCache` holds three dictionaries, keyed by `(Symbol, size)` tuples:

| Storage | Key type | Value type |
|:--------|:---------|:-----------|
| Matrices | `(Symbol, (nrows, ncols))` | `Matrix{Float64}` |
| Vectors | `(Symbol, length)` | `Vector{Float64}` |
| Lower-triangular | `(Symbol, dim)` | `LowerTriangular{Float64}` |

Create a new cache with `GPCache()` (all dictionaries start empty; buffers are allocated on first access).

## Accessors

```@docs
getcache
getcache_lowermatrix
```

## In-place matrix operations

These functions use the cache to perform matrix multiplications without allocating intermediate arrays:

```@docs
mul_A_B!
mul_A_B_A!
mul_A_B_At!
mul_A_v!
```
