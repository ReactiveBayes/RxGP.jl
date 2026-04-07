# Rule for GroupInverseRotation :T  —  EP message to pose
#
# Given vectors X (object frame) and Y (global frame) with  Y = Rᵀ X,
# infer the rotation R.  We solve  R Y = X  via SVD on  X Yᵀ  (Wahba problem).
#
# The outgoing message is a PoseBelief.  Because only rotation is observable
# here (no translational information from vector rotation), the translational
# components of the EP site will have very large covariance, leaving translation
# effectively unconstrained by this factor.

@rule GroupInverseRotation(:T, Marginalisation) (q_out::Any, q_in::Any, q_T::PoseBelief, meta::GroupInverseRotationMeta) = begin
    N = getN(meta)
    μ_X, Σ_X = _ga_mean_cov(q_in)
    μ_Y, Σ_Y = _ga_mean_cov(q_out)

    # --- Initialise EP site on first iteration ---
    if meta.m_T_old === nothing
        meta.m_T_old = PoseBelief(q_T.Tmean, 1e6 * Matrix{Float64}(I, 6, 6))
    end

    # --- Cavity on pose ---
    qT_c = cavity_pose(q_T, meta.m_T_old; N=N)

    Xs = _sample_gaussian(μ_X, Σ_X, N)
    Ys = _sample_gaussian(μ_Y, Σ_Y, N)

    # --- SVD rotation recovery per sample ---
    # Relation: R Y = X  →  R = X Yᵀ (Yᵗ Y)⁻¹  best via SVD of X Yᵀ
    T_samples = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        X_mat = reshape(Xs[i], 3, :)    # 3×K  (object frame)
        Y_mat = reshape(Ys[i], 3, :)    # 3×K  (global frame)

        H = X_mat * Y_mat'              # R Y ≈ X  →  best R from X Yᵀ
        F = svd(H)
        d = sign(det(F.V * F.U'))
        S = Diagonal([1.0, 1.0, d])
        R = F.V * S * F.U'

        # Build full SE(3) matrix, keeping current translation prior mean
        T_i = Matrix{Float64}(I, 4, 4)
        T_i[1:3, 1:3] .= R
        T_i[1:3, 4]   .= q_T.Tmean[1:3, 4]   # translation not observable → keep prior
        T_samples[i] = T_i
    end

    # --- Fit new marginal from samples ---
    qT_new = fit_pose(T_samples)

    return qT_new
    # # --- EP site extraction: site = q_new / cavity ---
    # Tc_samples = sample_pose(qT_c, N)
    # ξc = [logmap(inv(qT_new.Tmean) * T) for T in Tc_samples]
    # Xc = hcat(ξc...)
    # Σ_c = cov(Xc')

    # Λ_new = inv(qT_new.Σ)
    # Λ_c   = inv(Σ_c + 1e-9I)

    # Λ_site = Λ_new - Λ_c
    # Λ_site = Symmetric((Λ_site + Λ_site') / 2)
    # evals, evecs = eigen(Λ_site)
    # evals = max.(evals, 1e-9)
    # Λ_site = evecs * Diagonal(evals) * evecs'

    # Σ_site = inv(Λ_site)

    # result = PoseBelief(qT_new.Tmean, Σ_site)
    # meta.m_T_old = result

    # return result
end
