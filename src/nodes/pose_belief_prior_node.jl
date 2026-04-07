# PoseBeliefPrior node — stochastic prior for a PoseBelief on SE(3)
#
# Factor graph interfaces:
#   out    : the random SE(3) pose variable  (PoseBelief)
#   Tmean : 4×4 mean transformation          (data / PointMass)
#   Σ      : 6×6 tangent-space covariance      (data / PointMass)
#
# Models a concentrated Gaussian on SE(3):
#   p(T | Tmean, Σ) ∝ exp( -½ ξᵀ Σ⁻¹ ξ ),   ξ = logmap(Tmean⁻¹ T)

export PoseBeliefPrior

struct PoseBeliefPrior end

@node PoseBeliefPrior Stochastic [out, Tmean, Σ]

# ---- Average energy:  ⟨-log p(T | Tmean, Σ)⟩_{q(T)} ---- #
@average_energy PoseBeliefPrior (q_out::PoseBelief, q_Tmean::PointMass, q_Σ::PointMass) = begin
    T_prior = mean(q_Tmean)
    Σ_prior = mean(q_Σ)
    Λ_prior = inv(Σ_prior)
    N = 200
    Ts = sample_pose(q_out, N)
    E = 0.0
    for i in 1:N
        ξ = logmap(inv(T_prior) * Ts[i])
        E += dot(ξ, Λ_prior, ξ)
    end
    n = size(Σ_prior, 1)  # 6
    return 0.5 * E / N + 0.5 * logdet(Σ_prior) + 0.5 * n * log(2π)
end
