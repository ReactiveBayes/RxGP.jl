# rule for "w" edge (univariate batch case)
@rule UniSGP_Batch(:w, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_θ::PointMass, meta::UniSGPMeta,) = begin
    θ = mean(q_θ)
    kernel = getKernel(meta)
    μ_v, Σ_v = mean_cov_vector_matrix(q_v)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    Rv = Σ_v + μ_v * μ_v'
    D = get_dims_data(meta)

    μ_y_batch, Σ_y_batch = mean_cov_vector_matrix(q_out)
    N_j = length(μ_y_batch)
    μ_in_flat = mean(q_in)
    is_dist_in = q_in isa Distribution
    Σ_in_flat = is_dist_in ? cov(q_in) : nothing

    I_sum = 0.0
    for j in 1:N_j
        idx = (j-1)*D+1 : j*D
        μ_y = μ_y_batch[j]
        Σ_y = Σ_y_batch[j,j]

        if is_dist_in
            μ_j = μ_in_flat[idx]; Σ_j = Σ_in_flat[idx, idx]
            q_j = D == 1 ? NormalMeanVariance(μ_j[1], Σ_j[1,1]) : MvNormalMeanCovariance(μ_j, Matrix(Σ_j))
            Ψx = approximate_kernel_expectation(meta.method, (x) -> [apply_mean_fn(x, mf)], q_j)[1]
            Ψxx = approximate_kernel_expectation(meta.method, (x) -> [apply_mean_fn(x, mf) * apply_mean_fn(x, mf)], q_j)[1]
            Ψ0 = approximate_kernel_expectation(meta.method,(x) -> kernelmatrix(kernel(θ), [x], [x]), q_j)[1]
            Ψ1_trans = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_j)
            Ψ2 = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_j) + 1e-8*I
            Ψ3 = approximate_kernel_expectation(meta.method, (x) -> apply_mean_fn(x, mf) * kernelmatrix(kernel(θ), [x], meta.Xu), q_j)
        else
            x_j = μ_in_flat[idx]
            Ψx = apply_mean_fn(x_j, mf)
            Ψxx = Ψx^2
            Ψ0 = kernelmatrix(kernel(θ), [x_j], [x_j])[1]
            Ψ1_trans = kernelmatrix(kernel(θ), meta.Xu, [x_j])
            Ψ2 = kernelmatrix(kernel(θ), meta.Xu, [x_j]) * kernelmatrix(kernel(θ), [x_j], meta.Xu) + 1e-8*I
            Ψ3 = Ψx * kernelmatrix(kernel(θ), [x_j], meta.Xu)
        end

        Ψ2_Ku_mxu = Ψ2 * Ku_mxu

        I1 = Ψ0
        α = meta.KuuF.L \ Ψ1_trans
        I1 -= jdotavx(α, α)

        I4 = (
            μ_y^2
            + Σ_y
            - 2*μ_y*( Ψx + jdotavx(Ψ1_trans, (μ_v - Ku_mxu)) )
            + Ψxx
            + tr(Rv * Ψ2)
            + jdotavx(Ku_mxu, Ψ2_Ku_mxu)
            + 2*jdotavx(Ψ3, (μ_v - Ku_mxu))
            - 2*jdotavx(μ_v, Ψ2_Ku_mxu)
        )
        I4 = clamp(I4, 1e-12, 1e12)

        I_sum += I1 + I4
    end

    return GammaShapeRate(1 + 0.5*N_j, 0.5*I_sum)
end
