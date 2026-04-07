# Rule for GroupInverseRotation :out  —  forward message to Y
#
# Y = Rᵀ X  ⟹  sample T and X, apply inverse rotation, fit Gaussian.

@rule GroupInverseRotation(:out, Marginalisation) (q_in::Any, q_T::PoseBelief, meta::GroupInverseRotationMeta) = begin
    N = getN(meta)
    μ_X, Σ_X = _ga_mean_cov(q_in)

    Ts = sample_pose(q_T, N)
    Xs = _sample_gaussian(μ_X, Σ_X, N)

    Ys = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        R = Ts[i][1:3, 1:3]
        X = reshape(Xs[i], 3, :)
        Y = R' * X          # Rᵀ X
        Ys[i] = vec(Y)
    end

    μ_Y, Σ_Y = _fit_gaussian(Ys)
    return MvNormalMeanCovariance(μ_Y, Σ_Y)
end
