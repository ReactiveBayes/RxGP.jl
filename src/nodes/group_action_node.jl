# Group Action node:  Y = T ⊳ X   (SE(3) acting on ℝ³ⁿ)
#
# Factor graph interfaces:
#   out (Y) : observed / latent point-cloud in global frame  (ℝ³ⁿ vector)
#   in  (X) : point-cloud in object frame                     (ℝ³ⁿ vector)
#   T       : SE(3) pose                                      (PoseBelief)
#
# The deterministic relation is  Y = T * X  (each 3-column of X is transformed by T).
#
# Message passing uses Monte-Carlo sampling (MCS) for the Gaussian edges
# and MCS + EP cavity correction for the pose edge, following the prototype
# in SA__MCS_EP__CO_files/Group_Action.jl.

export GroupAction, GroupActionMeta

struct GroupAction end

@node GroupAction Deterministic [out, in, T]

# ---- Metadata ---- #
"""
    GroupActionMeta(; N_samples=200)

Metadata for the GroupAction node controlling the number of Monte-Carlo samples
used in the sampling-based message computations.
"""
mutable struct GroupActionMeta
    N_samples::Int                          # MC sample count
    m_T_old::Union{Nothing, PoseBelief}     # stored EP site for pose (previous iteration)
end

function GroupActionMeta(; N_samples::Int=200)
    return GroupActionMeta(N_samples, nothing)
end

getN(meta::GroupActionMeta) = meta.N_samples

# ---- Average energy ---- #
@average_energy GroupAction (q_out::Any, q_in::Any, q_T::PoseBelief, meta::GroupActionMeta) = begin
    N = getN(meta)
    μ_Y, Σ_Y = _ga_mean_cov(q_out)
    μ_X, Σ_X = _ga_mean_cov(q_in)

    Ts = sample_pose(q_T, N)
    Xs = _sample_gaussian(μ_X, Σ_X, N)
    Ys = _sample_gaussian(μ_Y, Σ_Y, N)

    E = 0.0
    for i in 1:N
        X = reshape(Xs[i], 3, :)
        Y_pred = _apply_pose(Ts[i], X)
        Y_true = reshape(Ys[i], 3, :)
        E += sum(abs2, Y_pred .- Y_true)
    end
    return E / N
end

# ====================================================================
# Internal helpers
# ====================================================================

"""Extract (mean_vector, covariance_matrix) from various belief types."""
function _ga_mean_cov(q)
    if q isa PoseBelief
        error("GroupAction edges in/out must be Euclidean, not PoseBelief")
    elseif q isa PointMass
        μ = vec(mean(q))
        return μ, zeros(length(μ), length(μ))
    else
        # MultivariateNormal or similar
        μ, Σ = mean_cov(q)
        return vec(μ), Matrix(Σ)
    end
end

"""Sample `N` vectors from 𝒩(μ, Σ)."""
function _sample_gaussian(μ::AbstractVector, Σ::AbstractMatrix, N::Int)
    L = cholesky(Symmetric(Σ + 1e-10I)).L
    return [μ + L * randn(length(μ)) for _ in 1:N]
end

"""Apply SE(3) matrix `T` (4×4) to each 3-column of X (3×K)."""
function _apply_pose(T::AbstractMatrix, X::AbstractMatrix)
    R = T[1:3, 1:3]
    t = T[1:3, 4]
    return R * X .+ t
end

"""Fit mean and covariance from a vector of sample vectors."""
function _fit_gaussian(samples::AbstractVector{<:AbstractVector})
    X = hcat(samples...)
    μ = vec(mean(X, dims=2))
    Σ = cov(X')
    return μ, Σ
end
