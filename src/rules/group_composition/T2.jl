# Rule for GroupComposition :T2  —  backward message:  T2 = inv(T1) * T3
#
# Uses EP cavity on T1 and T3, samples and inverts, fits T2, then extracts
# the EP site.

@rule GroupComposition(:T2, Marginalisation) (q_T1::PoseBelief, q_T3::PoseBelief, q_T2::PoseBelief, meta::GroupCompositionMeta) = begin
    N = getN(meta)

    # Lazy-init stored sites
    meta.m_T1_old === nothing && (meta.m_T1_old = _init_flat_site(q_T1.Tmean))
    meta.m_T2_old === nothing && (meta.m_T2_old = _init_flat_site(q_T2.Tmean))
    meta.m_T3_old === nothing && (meta.m_T3_old = _init_flat_site(q_T3.Tmean))

    # Cavities on T1 and T3
    p1c = cavity_pose(q_T1, meta.m_T1_old; N=N)
    p3c = cavity_pose(q_T3, meta.m_T3_old; N=N)

    # Sample from cavities: T2 = inv(T1) * T3
    T1s = sample_pose(p1c, N)
    T3s = sample_pose(p3c, N)
    T2s = [inv(T1s[i]) * T3s[i] for i in 1:N]

    # Fit new marginal for T2
    q2_new = fit_pose(T2s)

    # Cavity on T2 for site extraction
    q2c = cavity_pose(q_T2, meta.m_T2_old; N=N)

    # Extract EP site
    result = _ep_site_extract(q2_new, q2c, N)
    meta.m_T2_old = result
    return result
end
