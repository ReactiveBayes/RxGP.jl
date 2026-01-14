# rule for "out" (univariate case)
@rule UniSGP(:out, Marginalisation) (q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_w::Any, q_θ::PointMass, meta::UniSGPMeta,) = begin
    kernel = getKernel(meta)
    θ = mean(q_θ)
    μ_v = mean(q_v)
    Ψ1_trans = similar(meta.Ψ1_trans)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu

    if q_in isa Distribution
        Ψx = approximate_kernel_expectation(meta.method, (x) -> [apply_mean_fn(x, mf)], q_in)[1]
        approximate_kernel_expectation!(meta.Ψ1_trans, meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_in)
    else
        μ_in, Σ_in = mean_cov_vector_matrix(q_in)
        Ψx = apply_mean_fn(μ_in, mf)
        kernelmatrix!(meta.Ψ1_trans, kernel(θ), meta.Xu, [μ_in])
    end

    μ_y = Ψx + jdotavx(meta.Ψ1_trans, μ_v) - jdotavx(meta.Ψ1_trans, Ku_mxu)

    return NormalMeanPrecision(μ_y, mean(q_w))
end