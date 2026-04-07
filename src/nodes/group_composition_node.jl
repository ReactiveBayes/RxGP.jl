# Group Composition node:  T3 = T1 ∘ T2   (SE(3) × SE(3) → SE(3))
#
# Factor graph interfaces:
#   T3 : composed pose (output)
#   T1 : left  pose factor
#   T2 : right pose factor
#
# All three edges carry PoseBelief. Message passing uses MCS + EP cavity
# correction on all interfaces following the prototype in
# SA__MCS_EP__CO_files/Group_Composition.jl.

export GroupComposition, GroupCompositionMeta, avg_energy_comp

struct GroupComposition end

@node GroupComposition Deterministic [T3, T1, T2]

# ---- Metadata ---- #
"""
    GroupCompositionMeta(; N_samples=200)

Metadata for the GroupComposition node.
Stores EP site approximations for each interface and configures the MC sample count.
"""
mutable struct GroupCompositionMeta
    N_samples::Int
    m_T1_old::Union{Nothing, PoseBelief}
    m_T2_old::Union{Nothing, PoseBelief}
    m_T3_old::Union{Nothing, PoseBelief}
end

function GroupCompositionMeta(; N_samples::Int=200)
    return GroupCompositionMeta(N_samples, nothing, nothing, nothing)
end

getN(meta::GroupCompositionMeta) = meta.N_samples

# ---- Average energy ---- #
@average_energy GroupComposition (q_T3::PoseBelief, q_T1::PoseBelief, q_T2::PoseBelief, meta::GroupCompositionMeta) = begin
    N = getN(meta)
    T1s = sample_pose(q_T1, N)
    T2s = sample_pose(q_T2, N)
    T3s = sample_pose(q_T3, N)

    E = 0.0
    for i in 1:N
        T_pred  = T1s[i] * T2s[i]
        Δ = logmap(inv(T3s[i]) * T_pred)
        E += dot(Δ, Δ)
    end
    return E / N
end

# ===========================================================================
# Shared EP-site extraction helper
# ===========================================================================

"""
    _ep_site_extract(q_new::PoseBelief, q_cavity::PoseBelief, N::Int) -> PoseBelief

Given the new fitted marginal and the cavity, compute the EP site approximation
(i.e. site = q_new / cavity) in the tangent space of q_new, with PSD clamping.
"""
function _ep_site_extract(q_new::PoseBelief, q_cavity::PoseBelief, N::Int)
    Tc_samples = sample_pose(q_cavity, N)
    ξc = [logmap(inv(q_new.Tmean) * T) for T in Tc_samples]
    Xc = hcat(ξc...)
    Σ_c = cov(Xc')

    Λ_new = inv(q_new.Σ)
    Λ_c   = inv(Σ_c + 1e-9I)

    Λ_site = Λ_new - Λ_c
    Λ_site = Symmetric((Λ_site + Λ_site') / 2)
    evals, evecs = eigen(Λ_site)
    evals = max.(evals, 1e-9)
    Λ_site = evecs * Diagonal(evals) * evecs'

    Σ_site = inv(Λ_site)
    return PoseBelief(q_new.Tmean, Σ_site)
end

"""
    _init_flat_site(Tmean) -> PoseBelief

Create a very-wide (non-informative) EP site initialisation.
"""
function _init_flat_site(Tmean)
    return PoseBelief(Tmean, 1e6 * Matrix{Float64}(I, 6, 6))
end

"""
    avg_energy_comp(p1::PoseBelief, p2::PoseBelief, p3::PoseBelief; N=200) -> Float64

Standalone average-energy computation for testing:
E[‖log(T3⁻¹ T1 T2)‖²] under the given pose beliefs.
"""
function avg_energy_comp(p1::PoseBelief, p2::PoseBelief, p3::PoseBelief; N::Int=200)
    T1s = sample_pose(p1, N)
    T2s = sample_pose(p2, N)
    T3s = sample_pose(p3, N)
    E = 0.0
    for i in 1:N
        T_pred = T1s[i] * T2s[i]
        Δ = logmap(inv(T3s[i]) * T_pred)
        E += dot(Δ, Δ)
    end
    return E / N
end
