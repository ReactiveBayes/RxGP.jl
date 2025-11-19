# rule for "Wg" edge (univariate gradient case)
@rule UniSGP_Grad(:Wg, Marginalisation) (q_out::IN_OUT, q_in::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_θ::PointMass, meta::UniSGPMeta,) = begin
    θ = mean(q_θ)
    μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
    μ_in, Σ_in = mean_cov_vector_matrix(q_in)
    μ_v, Σ_v = mean_cov(q_v)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    D = get_dims_data(meta)
    Rv = μ_v * transpose(μ_v) + Σ_v
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    if q_in isa Distribution
        Ωx = approximate_kernel_expectation(meta.method, (x) -> Ex(x), q_in) + 1e-8*I
        Ω0 = approximate_kernel_expectation(meta.method, (x) -> Dxθ(x, θ), q_in) + 1e-8*I
        Ω1 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu), q_in) + 1e-8*I
        Ω2 = approximate_kernel_expectation(meta.method, (x) -> transpose(Cxθ_Xu(x, θ, meta.Xu))*Cxθ_Xu(x, θ, meta.Xu), q_in) + 1e-8*I
        Ω5 = approximate_kernel_expectation(meta.method, (x) -> Ex(x) * transpose(Ex(x)), q_in) + 1e-8*I
        Ω6 = approximate_kernel_expectation(meta.method, (x) -> Ex(x) * transpose(μ_v) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_in) + 1e-8*I
        Ω7 = approximate_kernel_expectation(meta.method, (x) -> Ex(x) * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_in) + 1e-8*I
        Ω8 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * μ_v * transpose(Ex(x)), q_in) + 1e-8*I
        Ω9 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * Rv * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_in) + 1e-8*I
        Ω10 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * μ_v * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_in) + 1e-8*I
        Ω11 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * Ku_mxu * transpose(Ex(x)), q_in) + 1e-8*I
        Ω12 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * Ku_mxu * transpose(μ_v) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_in) + 1e-8*I
        Ω13 = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ, meta.Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, meta.Xu)), q_in) + 1e-8*I
    else
        Ωx = Ex(μ_in)
        Ω0 = Dxθ(μ_in, θ)
        Ω1 = Cxθ_Xu(μ_in, θ, meta.Xu)
        Ω2 = transpose(Ω1) * Ω1
        Ω5 = Ωx * transpose(Ωx)
        Ω6 = Ωx * transpose(μ_v) * transpose(Ω1)
        Ω7 = Ωx * transpose(Ku_mxu) * transpose(Ω1)
        Ω8 = Ω1 * μ_v * transpose(Ωx)
        Ω9 = Ω1 * Rv * transpose(Ω1)
        Ω10 = Ω1 * μ_v * transpose(Ku_mxu) * transpose(Ω1)
        Ω11 = Ω1 * Ku_mxu * transpose(Ωx)
        Ω12 = Ω1 * Ku_mxu * transpose(μ_v) * transpose(Ω1)
        Ω13 = Ω1 * Ku_mxu * transpose(Ku_mxu) * transpose(Ω1)
    end


    G1 = Ω0 - Ω1 * (meta.KuuF \ transpose(Ω1))
    A_G2 = typeof(q_out) <: MultivariateNormalDistributionsFamily ? μ_ω * transpose(μ_ω) + Σ_ω : μ_ω * transpose(μ_ω)
    B_G2 = μ_ω * ( transpose(Ωx) + ( transpose(μ_v) - transpose(Ku_mxu) ) * transpose(Ω1) )
    C_G2 = (Ωx + Ω1 * ( μ_v - Ku_mxu )) * transpose(μ_ω)
    D_G2 = Ω5 + Ω6 - Ω7 + Ω8 + Ω9 - Ω10 - Ω11 - Ω12 + Ω13
    G2 = A_G2 - B_G2 - C_G2 + D_G2

    n_Wg = D + 2
    inv_V_Wg = G1 + G2

    return WishartFast(n_Wg, inv_V_Wg)
end