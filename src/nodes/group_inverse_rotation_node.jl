# Group Inverse Rotation node:  Y = R⁻¹ X  =  Rᵀ X
#
# Factor graph interfaces:
#   out (Y) : rotated vectors in the global frame        (ℝ³ⁿ vector)
#   in  (X) : vectors in the local object frame           (ℝ³ⁿ vector)
#   T       : SE(3) pose whose rotation R maps global→local  (PoseBelief)
#
# If T = [R t; 0 1] transforms *points* from global to object-local,
# then Rᵀ rotates *vectors* (normals, gradients) from object-local back
# to the global frame.  Translation is not applied (these are free vectors).
#
# Message passing mirrors GroupAction but uses Rᵀ in place of the full
# SE(3) action and operates on 3-vectors without homogeneous coordinates.

export GroupInverseRotation, GroupInverseRotationMeta

struct GroupInverseRotation end

@node GroupInverseRotation Deterministic [out, in, T]

# ---- Metadata ---- #
"""
    GroupInverseRotationMeta(; N_samples=200)

Metadata for the GroupInverseRotation node.
"""
mutable struct GroupInverseRotationMeta
    N_samples::Int
    m_T_old::Union{Nothing, PoseBelief}     # stored EP site for pose
end

function GroupInverseRotationMeta(; N_samples::Int=200)
    return GroupInverseRotationMeta(N_samples, nothing)
end

getN(meta::GroupInverseRotationMeta) = meta.N_samples

# ---- Average energy ---- #
@average_energy GroupInverseRotation (q_out::Any, q_in::Any, q_T::PoseBelief, meta::GroupInverseRotationMeta) = begin
    N = getN(meta)
    μ_Y, Σ_Y = _ga_mean_cov(q_out)
    μ_X, Σ_X = _ga_mean_cov(q_in)

    Ts = sample_pose(q_T, N)
    Xs = _sample_gaussian(μ_X, Σ_X, N)
    Ys = _sample_gaussian(μ_Y, Σ_Y, N)

    E = 0.0
    for i in 1:N
        R = Ts[i][1:3, 1:3]
        X = reshape(Xs[i], 3, :)
        Y_pred = R' * X            # Rᵀ X
        Y_true = reshape(Ys[i], 3, :)
        E += sum(abs2, Y_pred .- Y_true)
    end
    return E / N
end
