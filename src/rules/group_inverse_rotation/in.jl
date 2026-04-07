# Rule for GroupInverseRotation :in  —  backward message to X
#
# X = R Y  (since Y = Rᵀ X  ⟹  X = R Y).
# Sample T and Y, rotate forward, fit Gaussian.

@rule GroupInverseRotation(:in, Marginalisation) (q_out::Any, q_T::PoseBelief, meta::GroupInverseRotationMeta) = begin
    N = getN(meta)
    μ_Y, Σ_Y = _ga_mean_cov(q_out)

    Ts = sample_pose(q_T, N)
    Ys = _sample_gaussian(μ_Y, Σ_Y, N)

    Xs = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        R = Ts[i][1:3, 1:3]
        Y = reshape(Ys[i], 3, :)
        X = R * Y            # invert of Rᵀ is R
        Xs[i] = vec(X)
    end

    μ_X, Σ_X = _fit_gaussian(Xs)
    return MvNormalMeanCovariance(μ_X, Σ_X)
end
