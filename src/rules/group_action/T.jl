# # This file implements the marginalization rule for the T group action using Monte Carlo sampling and Gauss–Newton updates in the tangent space.
@rule GroupAction(:T, Marginalisation) (
    q_out::Any,
    q_in::Any,
    # q_T::PoseBelief,
    meta::GroupActionMeta
) = begin

    N = getN(meta)

    # --- Extract joint Gaussians (stacked points) ---
    μ_X, Σ_X = _ga_mean_cov(q_in)
    μ_Y, Σ_Y = _ga_mean_cov(q_out)

    X̄ = reshape(μ_X, 3, :)
    Ȳ = reshape(μ_Y, 3, :)
    n_pts = size(X̄, 2)

    # --- Current pose ---
    T̄ = q_T.Tmean

    # --- Sample pose perturbations ---
    ξ_samples = _sample_gaussian(zeros(6), q_T.Σ, N)

    ξ_post = Vector{Vector{Float64}}(undef, N)

    for i in 1:N
        ξ = ξ_samples[i]

        # Pose sample
        T_i = T̄ * expmap(ξ)
        R = T_i[1:3, 1:3]
        t = T_i[1:3, 4]

        # --- Sample X, Y jointly ---
        Xs = reshape(_sample_gaussian(μ_X, Σ_X, 1)[1], 3, :)
        Ys = reshape(_sample_gaussian(μ_Y, Σ_Y, 1)[1], 3, :)

        # --- Forward prediction ---
        Y_pred = R * Xs .+ t

        # Residual (stacked)
        r = vec(Ys - Y_pred)

        # --- Build joint Jacobian ---
        J = zeros(3n_pts, 6)
        for j in 1:n_pts
            x = Xs[:, j]
            Rx = R * x
            J_block = hcat(-skew(Rx), I(3))
            J[3(j-1)+1:3j, :] .= J_block
        end

        # --- One-step Gauss–Newton update ---
        ξ_hat = (J'J + 1e-9I) \ (J' * r)

        ξ_post[i] = ξ_hat
    end

    # --- Fit Gaussian in tangent space ---
    Ξ = hcat(ξ_post...)
    μ_ξ = mean(Ξ, dims=2)
    Σ_ξ = cov(Ξ')

    μ_ξ = vec(μ_ξ)
    Σ_ξ += 1e-9I

    # --- Map back to pose ---
    T_new = T̄ * expmap(μ_ξ)
    qT_new = PoseBelief(T_new, Σ_ξ)

    return qT_new
end