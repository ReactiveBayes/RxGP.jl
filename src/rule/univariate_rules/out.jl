# rule for "out" (univariate case)
@rule UniSGP(:out, Marginalisation) (q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_w::Any, q_θ::PointMass, meta::UniSGPMeta,) = begin
    kernel = getKernel(meta)
    θ = mean(q_θ)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    μ_v = mean(q_v)
    Ψ1_trans = similar(meta.Ψ1_trans)
    mf = getMeanFn(meta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(meta.Xu, mf)
    if q_in isa Distribution
        approximate_kernel_expectation!(meta.Ψ1_trans, meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_in)
    else
        kernelmatrix!(meta.Ψ1_trans, kernel(θ), meta.Xu, [μ_in])
    end
    Ku_mxu = meta.KuuF \ mxu
    μ_y = mx + jdotavx(meta.Ψ1_trans, μ_v) - jdotavx(meta.Ψ1_trans, Ku_mxu)
    return NormalMeanPrecision(μ_y, mean(q_w))
end