# PoseBelief type and helpers for SE(3) pose estimation via message passing
#
# A PoseBelief represents a distribution over SE(3) via a mean transformation
# Tmean (4×4 matrix) and a covariance Σ (6×6) in the tangent space at Tmean
# following the concentrated-Gaussian-on-Lie-group convention of Barfoot (2017).

export PoseBelief, sample_pose, fit_pose, cavity_pose

import Base: ==

"""
    PoseBelief(Tmean, Σ)

A belief (approximate distribution) on SE(3).

- `Tmean` : 4×4 mean transformation matrix in SE(3).
- `Σ`      : 6×6 covariance in the tangent space at Tmean.
"""
struct PoseBelief{T<:AbstractMatrix, S<:AbstractMatrix}
    Tmean::T
    Σ::S
end

function ==(a::PoseBelief, b::PoseBelief)
    return a.Tmean == b.Tmean && a.Σ == b.Σ
end

BayesBase.vague(::Type{<:PoseBelief}) =
    PoseBelief(I(4), 1e6 * I(6))  # Large covariance in tangent space for vague prior

"""
    sample_pose(p::PoseBelief, N::Int) -> Vector{Matrix{Float64}}

Draw `N` samples from the concentrated Gaussian on SE(3) defined by `p`.
"""
function sample_pose(p::PoseBelief, N::Int)
    L = cholesky(Symmetric(p.Σ + 1e-10I)).L
    Ts = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        δ = L * randn(6)
        Ts[i] = p.Tmean * expmap(δ)
    end
    return Ts
end

"""
    fit_pose(Ts; maxiter=20) -> PoseBelief

Fit a PoseBelief (Fréchet mean + tangent covariance) from a collection of SE(3) samples
using iterative tangent-space averaging.
"""
function fit_pose(Ts::AbstractVector; maxiter::Int=20)
    T̄ = Ts[1]
    for _ in 1:maxiter
        ξs = [logmap(inv(T̄) * T) for T in Ts]
        μ = mean(hcat(ξs...), dims=2) |> vec
        T̄ = T̄ * expmap(μ)
        norm(μ) < 1e-8 && break
    end
    ξs = [logmap(inv(T̄) * T) for T in Ts]
    X = hcat(ξs...)
    Σ = cov(X')
    return PoseBelief(T̄, Σ)
end

"""
    cavity_pose(q::PoseBelief, m_old::PoseBelief; N=200) -> PoseBelief

Approximate EP cavity distribution on SE(3):  cavity = q / m_old.

Computed by projecting both into a common tangent space and subtracting
natural parameters, with eigenvalue clamping for numerical stability.
"""
function cavity_pose(q::PoseBelief, m_old::PoseBelief; N::Int=200)
    # Project m_old into tangent space at q.Tmean
    Ts = sample_pose(m_old, N)
    ξs = [logmap(inv(q.Tmean) * T) for T in Ts]
    X = hcat(ξs...)
    Σ_m = cov(X')

    # Natural parameter subtraction in tangent space
    Λ_q = inv(q.Σ)
    Λ_m = inv(Σ_m + 1e-9I)
    Λ_c = Λ_q - Λ_m
    Λ_c = Symmetric((Λ_c + Λ_c') / 2)

    # Clamp eigenvalues to ensure PSD
    evals, evecs = eigen(Λ_c)
    evals = max.(evals, 1e-9)
    Λ_c = evecs * Diagonal(evals) * evecs'

    Σ_c = inv(Λ_c)
    return PoseBelief(q.Tmean, Σ_c)
end

"""
    product_pose(p1::PoseBelief, p2::PoseBelief; N=200) -> PoseBelief

Approximate product of two PoseBeliefs by local Gaussian fusion in tangent space.
"""
function product_pose(p1::PoseBelief, p2::PoseBelief; N::Int=200)
    # Project p2 into tangent space of p1
    Ts = sample_pose(p2, N)
    ξs = [logmap(inv(p1.Tmean) * T) for T in Ts]
    X = hcat(ξs...)
    μ2_local = vec(mean(X, dims=2))
    Σ2_local = cov(X')

    # Combine Gaussians in tangent space (p1 has mean at origin in its own tangent)
    Λ1 = inv(p1.Σ)
    Λ2 = inv(Σ2_local + 1e-9I)
    Σ_new = inv(Λ1 + Λ2)
    μ_new = Σ_new * (Λ2 * μ2_local)   # Λ1 * 0 + Λ2 * μ2

    T_new = p1.Tmean * expmap(μ_new)
    return PoseBelief(T_new, Σ_new)
end

# ── ReactiveMP product rule dispatch ──────────────────────────────────────────
# PoseBelief is not a subtype of Distribution, so we dispatch directly on
# GenericProd (the fallback strategy in BayesBase).  Both orderings are handled.

function ReactiveMP.prod(::GenericProd, left::PoseBelief, right::PoseBelief)
    return product_pose(left, right)
end
