# # This file implements the marginalization rule for the T group action using the Unscented Transform (UT).
# @rule GroupAction(:T, Marginalisation) (
#     q_out::Any,
#     q_in::Any,
#     # q_T::PoseBelief,
#     meta::GroupActionMeta
# ) = begin

#     # --- Extract joint Gaussians ---
#     μ_X, Σ_X = _ga_mean_cov(q_in)
#     μ_Y, Σ_Y = _ga_mean_cov(q_out)

#     X̄ = reshape(μ_X, 3, :)
#     Ȳ = reshape(μ_Y, 3, :)
#     n_pts = size(X̄, 2)

#     # --- Current pose ---
#     T̄ = q_T.Tmean
#     Σ_T = q_T.Σ

#     # --- UT parameters ---
#     d = 6
#     α = 1e-2
#     κ = 0.0
#     β = 2.0

#     λ = α^2 * (d + κ) - d

#     # --- Sigma points ---
#     S = cholesky((d + λ) * Σ_T).L

#     ξ_sigma = Vector{Vector{Float64}}(undef, 2d + 1)
#     w_m = zeros(2d + 1)
#     w_c = zeros(2d + 1)

#     ξ_sigma[1] = zeros(6)
#     w_m[1] = λ / (d + λ)
#     w_c[1] = w_m[1] + (1 - α^2 + β)

#     for i in 1:d
#         ξ_sigma[i+1]     = S[:, i]
#         ξ_sigma[i+1+d]   = -S[:, i]

#         w_m[i+1]   = 1 / (2(d + λ))
#         w_m[i+1+d] = 1 / (2(d + λ))

#         w_c[i+1]   = w_m[i+1]
#         w_c[i+1+d] = w_m[i+1+d]
#     end

#     # --- Storage for mapped sigma points ---
#     ξ_out = Vector{Vector{Float64}}(undef, 2d + 1)

#     for i in 1:(2d + 1)
#         ξ = ξ_sigma[i]

#         # Pose perturbation
#         T_i = T̄ * expmap(ξ)
#         R = T_i[1:3, 1:3]
#         t = T_i[1:3, 4]

#         # --- Residual mean ---
#         μ_r = vec(Ȳ - (R * X̄ .+ t))

#         # --- Residual covariance ---
#         # Σ_r = Σ_Y + R Σ_X Rᵀ (blockwise)
#         Σ_r = copy(Σ_Y)

#         # Apply R Σ_X Rᵀ blockwise (3x3 blocks)
#         for j in 1:n_pts
#             idx = 3(j-1)+1:3j
#             Σ_r[idx, idx] += R * Σ_X[idx, idx] * R' # If Σ_X is block-diagonal (no cross-block terms)
#             # Σ_r += kron(I(n_pts), R) * Σ_X * kron(I(n_pts), R') # If Σ_X has cross-block terms, i.e. If Σ_X is full
#         end

#         # --- Build Jacobian ---
#         J = zeros(3n_pts, 6)
#         for j in 1:n_pts
#             x = X̄[:, j]
#             Rx = R * x
#             J_block = hcat(-skew(Rx), I(3))
#             J[3(j-1)+1:3j, :] .= J_block
#         end

#         # --- Weighted Gauss–Newton step ---
#         Λ = J' * (Σ_r \ J)
#         η = J' * (Σ_r \ μ_r)

#         ξ_hat = (Λ + 1e-9I) \ η

#         ξ_out[i] = ξ_hat
#     end

#     # --- UT mean ---
#     μ_ξ = zeros(6)
#     for i in 1:(2d + 1)
#         μ_ξ += w_m[i] * ξ_out[i]
#     end

#     # --- UT covariance ---
#     Σ_ξ = zeros(6, 6)
#     for i in 1:(2d + 1)
#         δ = ξ_out[i] - μ_ξ
#         Σ_ξ += w_c[i] * (δ * δ')
#     end

#     Σ_ξ += 1e-9I

#     # --- Map back to pose ---
#     T_new = T̄ * expmap(μ_ξ)

#     return PoseBelief(T_new, Σ_ξ)
# end