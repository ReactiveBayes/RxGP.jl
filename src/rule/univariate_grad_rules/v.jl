# rule for "v" edge (univariate gradient case)
@rule UniSGP_Grad(:v, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_Wg::NOISE_Wg, q_θ::PointMass, meta::UniSGPMeta) = begin
    θ = mean(q_θ)
    μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)

    Wg_bar = mean(q_Wg); @assert maximum(abs, Wg_bar - transpose(Wg_bar)) < 1e-8 "Wg_bar is not symmetric"
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    Ex = getEx(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    if q_in isa Distribution
        Ω1 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu), q_in)
        Ω3 = approximate_kernel_expectation(meta.method, (x) -> transpose(Cxθ_Xu(x, θ, meta.Xu))*Wg_bar*Cxθ_Xu(x, θ, meta.Xu), q_in)
        Ω4 = approximate_kernel_expectation(meta.method, (x) -> transpose(Ex(x))*Wg_bar*Cxθ_Xu(x, θ, meta.Xu), q_in)
    else
        Ω1 = Cxθ_Xu(μ_in, θ, meta.Xu)
        Ω3 = transpose(Ω1) * Wg_bar * Ω1
        Ω4 = transpose(Ex(μ_in)) * Wg_bar * Ω1
    end

    W_v_raw = Ω3 + 1e-8I
    W_v = Symmetric(W_v_raw, :U)  # trust upper triangle, zero-cost wrapper
    ξ_v = vec(Ω3 * Ku_mxu + transpose(Ω1) * Wg_bar * μ_ω - transpose(Ω4))

    # return BufferUniSGP(MvNormalWeightedMeanPrecision(ξ_v, W_v), meta)
    return MvNormalWeightedMeanPrecision(ξ_v, W_v)
end