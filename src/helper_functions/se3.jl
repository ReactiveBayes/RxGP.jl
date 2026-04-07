# SE(3) Lie group helpers for pose-related factor nodes
# Provides expmap, logmap, and supporting operations for group action / composition nodes.
#
# Convention (Barfoot, "State Estimation for Robotics"):
#   ξ ∈ ℝ⁶ = [ρ; θ]  where  ρ ∈ ℝ³ (translation part),  θ ∈ ℝ³ (rotation part).
#   expmap(ξ) -> 4×4 SE(3) matrix
#   logmap(T) -> ξ ∈ ℝ⁶

export expmap, logmap, skew3, left_jacobian_SO3, inv_left_jacobian_SO3

"""
    skew3(v::AbstractVector) -> Matrix{Float64}

Hat map: ℝ³ → so(3).
"""
function skew3(v::AbstractVector)
    return [  0.0  -v[3]  v[2];
             v[3]   0.0  -v[1];
            -v[2]   v[1]  0.0 ]
end

"""
    left_jacobian_SO3(θ::AbstractVector) -> Matrix{Float64}

Left Jacobian of SO(3) evaluated at rotation vector θ (Barfoot eq. 7.77).
"""
function left_jacobian_SO3(θ::AbstractVector)
    ϕ = norm(θ)
    if ϕ < 1e-10
        return Matrix{Float64}(I, 3, 3)
    end
    S = skew3(θ)
    return I + ((1 - cos(ϕ)) / ϕ^2) * S + ((ϕ - sin(ϕ)) / ϕ^3) * (S * S)
end

"""
    inv_left_jacobian_SO3(θ::AbstractVector) -> Matrix{Float64}

Inverse of the left Jacobian of SO(3).
"""
function inv_left_jacobian_SO3(θ::AbstractVector)
    ϕ = norm(θ)
    if ϕ < 1e-10
        return Matrix{Float64}(I, 3, 3)
    end
    S = skew3(θ)
    return I - 0.5 * S + (1/ϕ^2 - (1 + cos(ϕ))/(2ϕ * sin(ϕ))) * (S * S)
end

"""
    expmap(ξ::AbstractVector) -> Matrix{Float64}

Exponential map: se(3) → SE(3).  ξ = [ρ; θ] ∈ ℝ⁶.
"""
function expmap(ξ::AbstractVector)
    ρ = ξ[1:3]
    θ = ξ[4:6]
    ϕ = norm(θ)
    if ϕ < 1e-10
        R = Matrix{Float64}(I, 3, 3)
    else
        u = θ / ϕ
        S = skew3(u)
        R = I + sin(ϕ) * S + (1 - cos(ϕ)) * (S * S)
    end
    t = left_jacobian_SO3(θ) * ρ
    T = Matrix{Float64}(undef, 4, 4)
    T[1:3, 1:3] .= R
    T[1:3, 4]   .= t
    T[4, 1:3]   .= 0.0
    T[4, 4]      = 1.0
    return T
end

"""
    logmap(T::AbstractMatrix) -> Vector{Float64}

Logarithmic map: SE(3) → se(3).  Returns ξ = [ρ; θ] ∈ ℝ⁶.
"""
function logmap(T::AbstractMatrix)
    R = T[1:3, 1:3]
    t = T[1:3, 4]
    # rotation angle
    cos_ϕ = clamp((tr(R) - 1) / 2, -1.0, 1.0)
    ϕ = acos(cos_ϕ)
    if ϕ < 1e-10
        θ = zeros(3)
    else
        θ = (ϕ / (2sin(ϕ))) * [R[3,2]-R[2,3]; R[1,3]-R[3,1]; R[2,1]-R[1,2]]
    end
    ρ = inv_left_jacobian_SO3(θ) * t
    return [ρ; θ]
end
