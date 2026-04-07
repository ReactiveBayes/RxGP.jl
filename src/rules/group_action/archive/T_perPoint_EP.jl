# # Rule for GroupAction :T  —  EP message to pose
# #
# # Given observations Y and object-frame points X, infer the pose T such that Y ≈ T * X.
# # Uses Kabsch (SVD) alignment per MC sample pair, then moment-matching + EP cavity
# # correction to produce the outgoing site message.
# #
# # Changes from the prototype (Group_Action.jl  msg_to_pose_EP):
# #   1. Increased default sample count (N) for lower-variance estimates (meta-configurable).
# #   2. Added deterministic-sign SVD reflection correction for robustness.
# #   3. Lazy initialisation of the EP site (m_T_old) on first call.

# @rule GroupAction(:T, Marginalisation) (q_out::Any, q_in::Any, q_T::PoseBelief, meta::GroupActionMeta) = begin
#     N = getN(meta)
#     μ_X, Σ_X = _ga_mean_cov(q_in)
#     μ_Y, Σ_Y = _ga_mean_cov(q_out)

#     # --- Initialise EP site on first iteration ---
#     if meta.m_T_old === nothing
#         meta.m_T_old = PoseBelief(q_T.Tmean, 1e6 * Matrix{Float64}(I, 6, 6))
#     end

#     # --- Cavity on pose (EP: remove old site) ---
#     qT_c = cavity_pose(q_T, meta.m_T_old; N=N)

#     # Sample X, Y from their marginals (no cavity needed for Euclidean edges)
#     Xs = _sample_gaussian(μ_X, Σ_X, N)
#     Ys = _sample_gaussian(μ_Y, Σ_Y, N)

#     # --- Kabsch alignment per sample pair ---
#     T_samples = Vector{Matrix{Float64}}(undef, N)
#     for i in 1:N
#         X_mat = reshape(Xs[i], 3, :)
#         Y_mat = reshape(Ys[i], 3, :)

#         μx = mean(X_mat, dims=2)
#         μy = mean(Y_mat, dims=2)
#         Xc = X_mat .- μx
#         Yc = Y_mat .- μy

#         H = Xc * Yc'
#         F = svd(H)
#         # Reflection correction (ensure proper rotation)
#         d = sign(det(F.V * F.U'))
#         S = Diagonal([1.0, 1.0, d])
#         R = F.V * S * F.U'
#         t = vec(μy - R * μx)

#         T_i = Matrix{Float64}(I, 4, 4)
#         T_i[1:3, 1:3] .= R
#         T_i[1:3, 4]   .= t
#         T_samples[i] = T_i
#     end

#     # --- Fit new marginal from samples ---
#     qT_new = fit_pose(T_samples)

#     # --- EP site extraction: site = q_new / cavity ---
#     Tc_samples = sample_pose(qT_c, N)
#     ξc = [logmap(inv(qT_new.Tmean) * T) for T in Tc_samples]
#     Xc = hcat(ξc...)
#     Σ_c = cov(Xc')

#     Λ_new = inv(qT_new.Σ)
#     Λ_c   = inv(Σ_c + 1e-9I)

#     Λ_site = Λ_new - Λ_c
#     Λ_site = Symmetric((Λ_site + Λ_site') / 2)
#     evals, evecs = eigen(Λ_site)
#     evals = max.(evals, 1e-9)
#     Λ_site = evecs * Diagonal(evals) * evecs'

#     Σ_site = inv(Λ_site)

#     result = PoseBelief(qT_new.Tmean, Σ_site)

#     # --- Store new site for next EP sweep ---
#     meta.m_T_old = result

#     return result
# end
