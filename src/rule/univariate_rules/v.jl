# rule for "v" edge (univariate case)
@rule UniSGP(:v, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_w::NOISE_w, q_θ::PointMass, meta::UniSGPMeta) = begin
    w_bar = mean(q_w)
    θ = mean(q_θ)
    μ_y, Σ_y = mean_cov_scalar_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    kernel = getKernel(meta)
    mf = getMeanFn(meta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(meta.Xu, mf)

    Ku_mxu = (meta.KuuL * transpose(meta.KuuL)) \ mxu
    Ψ1_trans = similar(meta.Ψ1_trans)
    kernelmatrix!(Ψ1_trans,kernel(θ),meta.Xu, [μ_in])

    W_v = similar(meta.Ψ2)
    mul!(W_v,Ψ1_trans,Ψ1_trans',w_bar,0) #W = w * Ψ1_trans * Ψ1_trans'
    mul!(meta.Ψ2,Ψ1_trans,Ψ1_trans') # Ψ2 = Ψ1_trans * Ψ1_trans'

    ξ_v = vec(w_bar * Ψ1_trans * (μ_y - mx) + w_bar * meta.Ψ2 * Ku_mxu)
    return BufferUniSGP(MvNormalWeightedMeanPrecision(ξ_v, W_v), meta)
end