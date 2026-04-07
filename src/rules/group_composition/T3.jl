# Rule for GroupComposition :T3  —  forward message:  T3 = T1 * T2
#
# Uses EP cavity on T1 and T2, samples and composes, fits T3, then extracts
# the EP site.
#
# Changes from prototype:
#   1. Factored EP site extraction into shared `_ep_site_extract`.
#   2. Uses lazy initialisation of stored EP sites.

@rule GroupComposition(:T3, Marginalisation) (q_T1::PoseBelief, q_T2::PoseBelief, q_T3::PoseBelief, meta::GroupCompositionMeta) = begin
    N = getN(meta)

    # Lazy-init stored sites
    meta.m_T1_old === nothing && (meta.m_T1_old = _init_flat_site(q_T1.Tmean))
    meta.m_T2_old === nothing && (meta.m_T2_old = _init_flat_site(q_T2.Tmean))
    meta.m_T3_old === nothing && (meta.m_T3_old = _init_flat_site(q_T3.Tmean))

    # Cavities on inputs
    p1c = cavity_pose(q_T1, meta.m_T1_old; N=N)
    p2c = cavity_pose(q_T2, meta.m_T2_old; N=N)

    # Sample from cavities and compose
    T1s = sample_pose(p1c, N)
    T2s = sample_pose(p2c, N)
    T3s = [T1s[i] * T2s[i] for i in 1:N]

    # Fit new marginal for T3
    q3_new = fit_pose(T3s)

    # Cavity on T3 for site extraction
    q3c = cavity_pose(q_T3, meta.m_T3_old; N=N)

    # Extract EP site
    result = _ep_site_extract(q3_new, q3c, N)
    meta.m_T3_old = result
    return result
end
