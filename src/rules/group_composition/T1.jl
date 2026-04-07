# Rule for GroupComposition :T1  —  backward message:  T1 = T3 * inv(T2)
#
# Uses EP cavity on T2 and T3, samples and inverts, fits T1, then extracts
# the EP site.

@rule GroupComposition(:T1, Marginalisation) (q_T2::PoseBelief, q_T3::PoseBelief, q_T1::PoseBelief, meta::GroupCompositionMeta) = begin
    N = getN(meta)

    # Lazy-init stored sites
    meta.m_T1_old === nothing && (meta.m_T1_old = _init_flat_site(q_T1.Tmean))
    meta.m_T2_old === nothing && (meta.m_T2_old = _init_flat_site(q_T2.Tmean))
    meta.m_T3_old === nothing && (meta.m_T3_old = _init_flat_site(q_T3.Tmean))

    # Cavities on T2 and T3
    p2c = cavity_pose(q_T2, meta.m_T2_old; N=N)
    p3c = cavity_pose(q_T3, meta.m_T3_old; N=N)

    # Sample from cavities: T1 = T3 * inv(T2)
    T2s = sample_pose(p2c, N)
    T3s = sample_pose(p3c, N)
    T1s = [T3s[i] * inv(T2s[i]) for i in 1:N]

    # Fit new marginal for T1
    q1_new = fit_pose(T1s)

    # Cavity on T1 for site extraction
    q1c = cavity_pose(q_T1, meta.m_T1_old; N=N)

    # Extract EP site
    result = _ep_site_extract(q1_new, q1c, N)
    meta.m_T1_old = result
    return result
end
