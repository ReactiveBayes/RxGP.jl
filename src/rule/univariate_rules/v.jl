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

    Ku_mxu = meta.KuuF \ mxu

    if q_in isa Distribution
        meta.Ψ1_trans = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_in)
        meta.Ψ2 = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in) + 1e-8*I
    else
        meta.Ψ1_trans = kernelmatrix(kernel(θ), meta.Xu, [μ_in])
        meta.Ψ2 = kernelmatrix(kernel(θ), meta.Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], meta.Xu) + 1e-8*I
    end

    W_v = w_bar * meta.Ψ2
    ξ_v = vec(w_bar * meta.Ψ1_trans * (μ_y - mx) + w_bar * meta.Ψ2 * Ku_mxu)

    return BufferUniSGP(MvNormalWeightedMeanPrecision(ξ_v, W_v), meta)
end