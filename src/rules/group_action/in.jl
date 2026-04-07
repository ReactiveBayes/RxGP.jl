# Rule for GroupAction :in  —  backward message to X
#
# X = T⁻¹ * Y  ⟹  sample T and Y, invert, fit Gaussian.
# No EP needed: X is a simple Euclidean variable.

@rule GroupAction(:in, Marginalisation) (q_out::Any, q_T::PoseBelief, meta::GroupActionMeta) = begin
    N = getN(meta)
    μ_Y, Σ_Y = _ga_mean_cov(q_out)

    Ts = sample_pose(q_T, N)
    Ys = _sample_gaussian(μ_Y, Σ_Y, N)

    Xs = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        Y = reshape(Ys[i], 3, :)
        X = _apply_pose(inv(Ts[i]), Y)
        Xs[i] = vec(X)
    end

    μ_X, Σ_X = _fit_gaussian(Xs)
    return MvNormalMeanCovariance(μ_X, Σ_X)
end
