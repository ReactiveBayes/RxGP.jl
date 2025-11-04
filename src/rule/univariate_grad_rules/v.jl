# rule for "v" edge (univariate gradient case)
@rule UniSGP_Grad(:v, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta) = begin
    θ = mean(q_θ)
    μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)

    Wg_bar = mean(q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ex = getEx(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    Ωx = Ex(μ_in)
    Ω1 = Cxθ_Xu(μ_in, θ, meta.Xu)

    Ku_mxu = (meta.KuuL * transpose(meta.KuuL)) \ mxu
    Ω3 = transpose(Ω1) * Wg_bar * Ω1
    Ω4 = transpose(Ωx) * Wg_bar * Ω1

    W_v = Ω3
    ξ_v = vec(Ω3 * Ku_mxu + transpose(Ω1) * Wg_bar * μ_ω - transpose(Ω4))

    return BufferUniSGP(MvNormalWeightedMeanPrecision(ξ_v, W_v), meta)
end