

@testsnippet univariate_grad_snippet begin
    using RxGP
    using RxInfer
    using ReactiveMP
    using Random
    using Distributions
    using StableRNGs
    using KernelFunctions
    using LinearAlgebra
    using StatsFuns
    using Test

    const grad_default_method = ReactiveMP.ghcubature(21)

    grad_mean_fn(x) = begin
        x_val = x isa Number ? x : x[1]
        x_val^2 + 0.35 * x_val - 0.1
    end

    function grad_fixture(; q_in=Normal(0.35, 0.4), q_out=MvNormalMeanCovariance([0.8], [0.25;;]), q_Wg=PointMass([1.4;;]), q_v=nothing, Xu=collect(range(-0.8, 0.8, length=5)))
        D = 1
        kernel, θ_val, _ = get_simple_kernel_and_params(D; kernel_spec=:SE)
        meta = get_GP_meta(D; method=grad_default_method, mean_fn=grad_mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=true, Xu=Xu, θ=θ_val)
        if q_v === nothing
            μ_v = collect(range(-0.5, 0.5, length=length(Xu)))
            Σ_v = 0.45 .* Matrix{Float64}(I, length(Xu), length(Xu))
            q_v = MvNormalMeanCovariance(μ_v, Σ_v)
        end
        return (; method=grad_default_method, D, Xu, Nu=length(Xu), kernel, θ_val, meta, q_in, q_out, q_v, q_Wg, q_θ=PointMass(θ_val))
    end
end

@testitem "node_rule/univariate_grad/Test GPMeta" setup=[univariate_grad_snippet] begin
    ctx = grad_fixture()
    meta = ctx.meta

    x_dummy = zeros(ctx.D)
    Ψ0 = kernelmatrix(ctx.kernel(ctx.θ_val), [x_dummy])[1]
    Ψ1_trans = kernelmatrix(ctx.kernel(ctx.θ_val), ctx.Xu, [x_dummy])
    Ψ2 = kernelmatrix(ctx.kernel(ctx.θ_val), ctx.Xu, [x_dummy]) * kernelmatrix(ctx.kernel(ctx.θ_val), [x_dummy], ctx.Xu)

    @test getInducingInput(meta) == ctx.Xu
    @test getKernel(meta) == ctx.kernel
    @test typeof(getKernel(meta)) <: Function
    @test getmethod(meta) == ctx.method
    @test getΨ0(meta) == Ψ0
    @test getΨ1_trans(meta) == Ψ1_trans
    @test getΨ2(meta) == Ψ2
    @test getcounter(meta) == 0
    @test getN(meta) == 1

    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Fxθ = getFxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    x_probe = 0.27
    expected_grad = 2 * x_probe + 0.35
    @test isapprox(Ex([x_probe])[1], expected_grad; atol=1e-10)
    Dx_val = Dxθ([x_probe], ctx.θ_val)
    @test size(Dx_val) == (ctx.D, ctx.D)
    @test all(isfinite, Dx_val)
    @test Fxθ([x_probe], ctx.θ_val) == zeros(1, ctx.D)
    @test size(Cxθ_Xu([x_probe], ctx.θ_val, ctx.Xu)) == (ctx.D, ctx.Nu)
end

@testitem "node_rule/univariate_grad/Test out rule" setup=[univariate_grad_snippet] begin
    ctx = grad_fixture()
    meta = ctx.meta
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    μ_v = mean(ctx.q_v)
    Wg_bar = mean(ctx.q_Wg)
    θ = ctx.θ_val
    Ex = getEx(meta)
    Cxθ_Xu = getCxθ_Xu(meta)

    slow_mean = begin
        Ωx = approximate_kernel_expectation(ctx.method, (x) -> Ex(x), ctx.q_in)
        Ω1 = approximate_kernel_expectation(ctx.method, (x) -> Cxθ_Xu(x, θ, ctx.Xu), ctx.q_in)
        Ωx + Ω1 * (μ_v - Ku_mxu)
    end

    msg = @call_rule UniSGP_Grad(:out, Marginalisation) (q_in = ctx.q_in, q_v = ctx.q_v, q_Wg = ctx.q_Wg, q_θ = ctx.q_θ, meta = meta)
    @test typeof(msg) <: MultivariateGaussianDistributionsFamily
    @test isapprox(mean(msg), vec(slow_mean); atol=1e-6)
    @test isapprox(inv(cov(msg)), Wg_bar; atol=1e-6)

    det_ctx = grad_fixture(q_in = PointMass(0.12))
    meta_det = det_ctx.meta
    mf_det = getMeanFn(meta_det)
    mxu_det = apply_mean_fn.(meta_det.Xu, mf_det)
    Ku_mxu_det = meta_det.KuuF \ mxu_det
    μ_v_det = mean(det_ctx.q_v)
    Wg_bar_det = mean(det_ctx.q_Wg)
    μ_in = mean(det_ctx.q_in)
    Ωx_det = Ex(μ_in)
    Ω1_det = Cxθ_Xu(μ_in, det_ctx.θ_val, det_ctx.Xu)
    slow_det_mean = Ωx_det + Ω1_det * (μ_v_det - Ku_mxu_det)

    msg_det = @call_rule UniSGP_Grad(:out, Marginalisation) (q_in = det_ctx.q_in, q_v = det_ctx.q_v, q_Wg = det_ctx.q_Wg, q_θ = det_ctx.q_θ, meta = meta_det)
    @test typeof(msg_det) <: MultivariateGaussianDistributionsFamily
    @test isapprox(mean(msg_det), vec(slow_det_mean); atol=1e-6)
    @test isapprox(inv(cov(msg_det)), Wg_bar_det; atol=1e-6)
end

@testitem "node_rule/univariate_grad/Test in rule" setup=[univariate_grad_snippet] begin
    ctx = grad_fixture()
    meta = ctx.meta
    θ = ctx.θ_val
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    μ_v, Σ_v = mean_cov(ctx.q_v)
    μ_ω, _ = mean_cov_vector_matrix(ctx.q_out)
    Rv = μ_v * transpose(μ_v) + Σ_v
    Wg_bar = mean(ctx.q_Wg)

    Dx = (x) -> Dxθ(x, θ)
    Cx = (x) -> Cxθ_Xu(x, θ, ctx.Xu)
    Usolve = (M) -> meta.KuuF \ transpose(M)
    Qx = (x) -> Dx(x) - Cx(x) * Usolve(Cx(x))
    WC = (x) -> Wg_bar * Cx(x)
    CTWC = (x) -> transpose(Cx(x)) * Wg_bar * Cx(x)
    CTWC_Ku_mxu = (x) -> CTWC(x) * Ku_mxu
    slow_log = (x_val) -> begin
        x_vec = [x_val]
        part_A = tr(Wg_bar * Qx(x_vec))
        part_B = -2 * transpose(μ_ω) * Wg_bar * (Ex(x_vec) + Cx(x_vec) * (μ_v - Ku_mxu))
        part_C = (
            transpose(Ex(x_vec)) * Wg_bar * Ex(x_vec) +
            tr(Rv * CTWC(x_vec)) +
            (transpose(Ku_mxu) - 2 * transpose(μ_v)) * CTWC_Ku_mxu(x_vec) +
            2 * transpose(Ex(x_vec)) * WC(x_vec) * (μ_v - Ku_mxu)
        )
        -0.5 * (part_A + part_B + part_C)
    end

    msg = @call_rule UniSGP_Grad(:in, Marginalisation) (q_out = ctx.q_out, q_v = ctx.q_v, q_Wg = ctx.q_Wg, q_θ = ctx.q_θ, meta = meta)
    @test typeof(msg) <: ContinuousUnivariateLogPdf
    for x_probe in (0.1, 0.8, -0.4)
        @test isapprox(logpdf(msg, x_probe), slow_log(x_probe); atol=1e-6)
    end
end

@testitem "node_rule/univariate_grad/Test v rule" setup=[univariate_grad_snippet] begin
    ctx = grad_fixture()
    meta = ctx.meta
    θ = ctx.θ_val
    Ex = getEx(meta)
    Cxθ_Xu = getCxθ_Xu(meta)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    μ_ω, _ = mean_cov_vector_matrix(ctx.q_out)
    Wg_bar = mean(ctx.q_Wg)

    Ω1 = approximate_kernel_expectation(ctx.method, (x) -> Cxθ_Xu(x, θ, ctx.Xu), ctx.q_in)
    Ω3 = approximate_kernel_expectation(ctx.method, (x) -> transpose(Cxθ_Xu(x, θ, ctx.Xu)) * Wg_bar * Cxθ_Xu(x, θ, ctx.Xu), ctx.q_in)
    Ω4 = approximate_kernel_expectation(ctx.method, (x) -> transpose(Ex(x)) * Wg_bar * Cxθ_Xu(x, θ, ctx.Xu), ctx.q_in)
    ξ = vec(Ω3 * Ku_mxu + transpose(Ω1) * Wg_bar * μ_ω - transpose(Ω4))
    W_v = Ω3 + 1e-8*I
    gt_cov = cholinv(W_v)
    gt_mean = gt_cov * ξ

    msg = @call_rule UniSGP_Grad(:v, Marginalisation) (q_out = ctx.q_out, q_in = ctx.q_in, q_Wg = ctx.q_Wg, q_θ = ctx.q_θ, meta = meta)
    @test typeof(msg) <: BufferUniSGP
    @test typeof(msg.qv) <: MultivariateGaussianDistributionsFamily
    @test isapprox(mean(msg.qv), gt_mean; atol=1e-6)
    @test isapprox(cov(msg.qv), gt_cov; atol=1e-6)

    det_ctx = grad_fixture(q_in = PointMass(0.2), q_out = MvNormalMeanCovariance([0.5], [0.0;;]))
    meta_det = det_ctx.meta
    mf_det = getMeanFn(meta_det)
    mxu_det = apply_mean_fn.(meta_det.Xu, mf_det)
    Ku_mxu_det = meta_det.KuuF \ mxu_det
    μ_ω_det, _ = mean_cov_vector_matrix(det_ctx.q_out)
    Wg_bar_det = mean(det_ctx.q_Wg)
    μ_in = mean(det_ctx.q_in)
    Ω1_det = Cxθ_Xu(μ_in, det_ctx.θ_val, det_ctx.Xu)
    Ω3_det = transpose(Ω1_det) * Wg_bar_det * Ω1_det
    Ω4_det = transpose(Ex(μ_in)) * Wg_bar_det * Ω1_det
    ξ_det = vec(Ω3_det * Ku_mxu_det + transpose(Ω1_det) * Wg_bar_det * μ_ω_det - transpose(Ω4_det))
    W_det = Ω3_det + 1e-8*I

    msg_det = @call_rule UniSGP_Grad(:v, Marginalisation) (q_out = det_ctx.q_out, q_in = det_ctx.q_in, q_Wg = det_ctx.q_Wg, q_θ = det_ctx.q_θ, meta = meta_det)
    @test isapprox(mean(msg_det.qv), cholinv(W_det) * ξ_det; atol=1e-6)
    @test isapprox(cov(msg_det.qv), cholinv(W_det); atol=1e-6)
end

@testitem "node_rule/univariate_grad/Test Wg rule" setup=[univariate_grad_snippet] begin
    ctx = grad_fixture()
    meta = ctx.meta
    θ = ctx.θ_val
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    μ_v, Σ_v = mean_cov(ctx.q_v)
    μ_ω, Σ_ω = mean_cov_vector_matrix(ctx.q_out)
    Rv = μ_v * transpose(μ_v) + Σ_v

    expect = (fn) -> approximate_kernel_expectation(ctx.method, fn, ctx.q_in)
    Ωx = expect((x) -> Ex(x))
    Ω0 = expect((x) -> Dxθ(x, θ))
    Ω1 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu))
    Ω5 = expect((x) -> Ex(x) * transpose(Ex(x)))
    Ω6 = expect((x) -> Ex(x) * transpose(μ_v) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω7 = expect((x) -> Ex(x) * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω8 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * μ_v * transpose(Ex(x)))
    Ω9 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * Rv * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω10 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * μ_v * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω11 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * Ku_mxu * transpose(Ex(x)))
    Ω12 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * Ku_mxu * transpose(μ_v) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω13 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))

    G1 = Ω0 - Ω1 * (meta.KuuF \ transpose(Ω1))
    A_G2 = μ_ω * transpose(μ_ω) + Σ_ω
    B_G2 = μ_ω * (transpose(Ωx) + (transpose(μ_v) - transpose(Ku_mxu)) * transpose(Ω1))
    C_G2 = (Ωx + Ω1 * (μ_v - Ku_mxu)) * transpose(μ_ω)
    D_G2 = Ω5 + Ω6 - Ω7 + Ω8 + Ω9 - Ω10 - Ω11 - Ω12 + Ω13
    G2 = A_G2 - B_G2 - C_G2 + D_G2

    msg = @call_rule UniSGP_Grad(:Wg, Marginalisation) (q_out = ctx.q_out, q_in = ctx.q_in, q_v = ctx.q_v, q_θ = ctx.q_θ, meta = meta)
    n_gt = ctx.D + 2
    V_gt = cholinv(G1 + G2)
    n_w, V_w = params(msg)
    @test isapprox(n_w, n_gt; atol=1e-6)
    @test isapprox(V_w, V_gt; atol=1e-6)

    det_ctx = grad_fixture(q_in = PointMass(0.13), q_out = MvNormalMeanCovariance([0.6], [0.0;;]))
    meta_det = det_ctx.meta
    mf_det = getMeanFn(meta_det)
    mxu_det = apply_mean_fn.(meta_det.Xu, mf_det)
    Ku_mxu_det = meta_det.KuuF \ mxu_det
    μ_v_det, Σ_v_det = mean_cov(det_ctx.q_v)
    μ_ω_det, Σ_ω_det = mean_cov_vector_matrix(det_ctx.q_out)
    Rv_det = μ_v_det * transpose(μ_v_det) + Σ_v_det
    Ωx_det = Ex(mean(det_ctx.q_in))
    Ω0_det = Dxθ(mean(det_ctx.q_in), det_ctx.θ_val)
    Ω1_det = Cxθ_Xu(mean(det_ctx.q_in), det_ctx.θ_val, det_ctx.Xu)
    Ω5_det = Ωx_det * transpose(Ωx_det)
    Ω6_det = Ωx_det * transpose(μ_v_det) * transpose(Ω1_det)
    Ω7_det = Ωx_det * transpose(Ku_mxu_det) * transpose(Ω1_det)
    Ω8_det = Ω1_det * μ_v_det * transpose(Ωx_det)
    Ω9_det = Ω1_det * Rv_det * transpose(Ω1_det)
    Ω10_det = Ω1_det * μ_v_det * transpose(Ku_mxu_det) * transpose(Ω1_det)
    Ω11_det = Ω1_det * Ku_mxu_det * transpose(Ωx_det)
    Ω12_det = Ω1_det * Ku_mxu_det * transpose(μ_v_det) * transpose(Ω1_det)
    Ω13_det = Ω1_det * Ku_mxu_det * transpose(Ku_mxu_det) * transpose(Ω1_det)
    G1_det = Ω0_det - Ω1_det * (meta_det.KuuF \ transpose(Ω1_det))
    A_G2_det = μ_ω_det * transpose(μ_ω_det) + Σ_ω_det
    B_G2_det = μ_ω_det * (transpose(Ωx_det) + (transpose(μ_v_det) - transpose(Ku_mxu_det)) * transpose(Ω1_det))
    C_G2_det = (Ωx_det + Ω1_det * (μ_v_det - Ku_mxu_det)) * transpose(μ_ω_det)
    D_G2_det = Ω5_det + Ω6_det - Ω7_det + Ω8_det + Ω9_det - Ω10_det - Ω11_det - Ω12_det + Ω13_det
    V_det = cholinv(G1_det + (A_G2_det - B_G2_det - C_G2_det + D_G2_det))

    msg_det = @call_rule UniSGP_Grad(:Wg, Marginalisation) (q_out = det_ctx.q_out, q_in = det_ctx.q_in, q_v = det_ctx.q_v, q_θ = det_ctx.q_θ, meta = meta_det)
    n_w_det, V_w_det = params(msg_det)
    @test isapprox(n_w_det, ctx.D + 2; atol=1e-6)
    @test isapprox(V_w_det, V_det; atol=1e-6)
end

@testitem "node_rule/univariate_grad/Test θ rule" setup=[univariate_grad_snippet] begin
    ctx = grad_fixture()
    meta = ctx.meta
    θ = ctx.θ_val
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    μ_v, Σ_v = mean_cov(ctx.q_v)
    μ_ω, _ = mean_cov_vector_matrix(ctx.q_out)
    Rv = μ_v * transpose(μ_v) + Σ_v
    Wg_bar = mean(ctx.q_Wg)

    Ω0 = (θ_val) -> approximate_kernel_expectation(ctx.method, (x) -> Dxθ(x, θ_val), ctx.q_in)
    Ω1 = (θ_val) -> approximate_kernel_expectation(ctx.method, (x) -> Cxθ_Xu(x, θ_val, ctx.Xu), ctx.q_in)
    Ω3 = (θ_val) -> approximate_kernel_expectation(ctx.method, (x) -> transpose(Cxθ_Xu(x, θ_val, ctx.Xu)) * Wg_bar * Cxθ_Xu(x, θ_val, ctx.Xu), ctx.q_in)
    Ω4 = (θ_val) -> approximate_kernel_expectation(ctx.method, (x) -> transpose(Ex(x)) * Wg_bar * Cxθ_Xu(x, θ_val, ctx.Xu), ctx.q_in)
    G1 = (θ_val) -> Ω0(θ_val) - Ω1(θ_val) * (meta.KuuF \ transpose(Ω1(θ_val)))
    part_A = (θ_val) -> 2 * dot(Ω4(θ_val), (μ_v - Ku_mxu)) + dot((transpose(Ku_mxu) - 2 * transpose(μ_v)), Ω3(θ_val) * Ku_mxu) + tr(Ω3(θ_val) * Rv)
    part_B = (θ_val) -> 2 * dot(transpose(μ_ω), Wg_bar * Ω1(θ_val) * (μ_v - Ku_mxu))
    slow_log = (θ_val) -> -0.5 * tr(Wg_bar * G1(θ_val)) - 0.5 * (part_A(θ_val) - part_B(θ_val))

    msg = @call_rule UniSGP_Grad(:θ, Marginalisation) (q_out = ctx.q_out, q_in = ctx.q_in, q_v = ctx.q_v, q_Wg = ctx.q_Wg, meta = meta)
    @test typeof(msg) <: ContinuousMultivariateLogPdf
    θ_shift_1 = θ .+ 0.05
    θ_shift_2 = θ .- 0.08
    @test isapprox(logpdf(msg, θ_shift_1), slow_log(θ_shift_1); atol=1e-7)
    @test isapprox(logpdf(msg, θ_shift_2), slow_log(θ_shift_2); atol=1e-7)
end

@testitem "node_rule/univariate_grad/Test average energy" setup=[univariate_grad_snippet] begin
    ctx = grad_fixture()
    meta = ctx.meta
    θ = ctx.θ_val
    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Ku_mxu = meta.KuuF \ mxu
    μ_v, Σ_v = mean_cov(ctx.q_v)
    μ_ω, Σ_ω = mean_cov_vector_matrix(ctx.q_out)
    Rv = μ_v * transpose(μ_v) + Σ_v
    Wg_bar = mean(ctx.q_Wg)
    E_logWg = mean(logdet, ctx.q_Wg)

    expect = (fn) -> approximate_kernel_expectation(ctx.method, fn, ctx.q_in)
    Ωx = expect((x) -> Ex(x))
    Ω0 = expect((x) -> Dxθ(x, θ))
    Ω1 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu))
    Ω5 = expect((x) -> Ex(x) * transpose(Ex(x)))
    Ω6 = expect((x) -> Ex(x) * transpose(μ_v) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω7 = expect((x) -> Ex(x) * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω8 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * μ_v * transpose(Ex(x)))
    Ω9 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * Rv * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω10 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * μ_v * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω11 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * Ku_mxu * transpose(Ex(x)))
    Ω12 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * Ku_mxu * transpose(μ_v) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))
    Ω13 = expect((x) -> Cxθ_Xu(x, θ, ctx.Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ, ctx.Xu)))

    G1 = Ω0 - Ω1 * (meta.KuuF \ transpose(Ω1))
    A_G2 = μ_ω * transpose(μ_ω) + Σ_ω
    B_G2 = μ_ω * (transpose(Ωx) + (transpose(μ_v) - transpose(Ku_mxu)) * transpose(Ω1))
    C_G2 = (Ωx + Ω1 * (μ_v - Ku_mxu)) * transpose(μ_ω)
    D_G2 = Ω5 + Ω6 - Ω7 + Ω8 + Ω9 - Ω10 - Ω11 - Ω12 + Ω13
    G2 = A_G2 - B_G2 - C_G2 + D_G2

    U_gt = 0.5 * tr(Wg_bar * (G1 + G2)) + (ctx.D / 2) * log(2π) - 0.5 * E_logWg

    marginals = (
        Marginal(ctx.q_out, false, false, nothing),
        Marginal(ctx.q_in, false, false, nothing),
        Marginal(ctx.q_v, false, false, nothing),
        Marginal(ctx.q_Wg, false, false, nothing),
        Marginal(ctx.q_θ, false, false, nothing)
    )

    U_node = score(AverageEnergy(), UniSGP_Grad, Val{(:out, :in, :v, :Wg, :θ)}(), marginals, meta)
    @test typeof(U_node) <: Float64
    @test isapprox(U_node, U_gt; atol=1e-6)
end