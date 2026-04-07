# Rule for GroupAction :out  —  forward message to Y
#
# Y = T * X  ⟹  sample T and X, propagate, fit Gaussian.
# No EP needed: Y is a simple Euclidean variable.

@rule GroupAction(:out, Marginalisation) (q_in::Any, q_T::PoseBelief, meta::GroupActionMeta) = begin
    N = getN(meta)
    μ_X, Σ_X = _ga_mean_cov(q_in)

    Ts = sample_pose(q_T, N)
    Xs = _sample_gaussian(μ_X, Σ_X, N)

    Ys = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        X = reshape(Xs[i], 3, :)
        Y = _apply_pose(Ts[i], X)
        Ys[i] = vec(Y)
    end

    μ_Y, Σ_Y = _fit_gaussian(Ys)
    return MvNormalMeanCovariance(μ_Y, Σ_Y)
end
