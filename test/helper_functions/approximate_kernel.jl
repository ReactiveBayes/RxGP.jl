@testsnippet setup_snippet begin
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

    const grad_default_method = ReactiveMP.ghcubature(50)

    test_mean_fn(x) = begin
        x_val = x isa Number ? x : x[1]
        x_val^2 + 0.35 * x_val - 0.1
    end

    function test_fixture(; q_x=Normal(0.35, 0.4), q_out=MvNormalMeanCovariance([0.8], [0.25;;]), q_Wg=PointMass([1.4;;]), q_v=nothing, Xu=collect(range(-0.8, 0.8, length=5)))
        D = 1
        kernel_spec = :SEn
        kernel, θ_val, _ = get_simple_kernel_and_params(D; kernel_spec=kernel_spec)
        meta = get_GP_meta(D; method=grad_default_method, mean_fn=test_mean_fn, kernel=kernel, kernel_spec=kernel_spec, mode=:AN, independent_SE_lengthscales=true, Xu=Xu, θ=θ_val)
        if q_v === nothing
            μ_v = collect(range(-0.5, 0.5, length=length(Xu)))
            Σ_v = 0.45 .* Matrix{Float64}(I, length(Xu), length(Xu))
            q_v = MvNormalMeanCovariance(μ_v, Σ_v)
        end
        return (; method=grad_default_method, D, Xu, Nu=length(Xu), kernel, θ_val, meta, q_x, q_out, q_v, q_Wg, q_θ=PointMass(θ_val))
    end
end

@testitem "helper_functions/approximate_kernel/approximate_kernel_expectation" setup=[setup_snippet] begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    ctx = test_fixture()
    method = ctx.method
    q_x = ctx.q_x

    # make sure we add new methods for approximate_kernel_expectation properly
    # make sure the function works as expected
    foo(x) = 2*x
    @test length(methods(approximate_kernel_expectation)) == length(methods(ReactiveMP.approximate_kernel_expectation))
    @test approximate_kernel_expectation(method,foo,q_x) == approximate_kernel_expectation(method, foo, mean(q_x),var(q_x))
    @test isapprox(approximate_kernel_expectation(method,foo,q_x),2*mean(q_x); atol=1e-9)
    @test approximate_kernel_expectation(GenUT(),foo,q_x) ≈ 2*mean(q_x)

    foo_2d(x) = [2*x, x]
    gbar = [1.5,0.5]
    @test approximate_kernel_expectation!(gbar, method, foo_2d, mean(q_x), var(q_x)) == gbar
    @test approximate_kernel_expectation!(gbar, method, foo_2d, q_x) == approximate_kernel_expectation!(gbar, method, foo_2d, mean(q_x), var(q_x))
    @test approximate_kernel_expectation(method,foo_2d,q_x) == approximate_kernel_expectation(method, foo_2d, mean(q_x),var(q_x))
    @test isapprox(approximate_kernel_expectation(method,foo_2d,q_x),[2*mean(q_x), mean(q_x)]; atol=1e-9)
    @test approximate_kernel_expectation(GenUT(),foo_2d,q_x) ≈ [2*mean(q_x), mean(q_x)]
end

@testitem "helper_functions/approximate_kernel/Psi approximation" setup=[setup_snippet] begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    rng = StableRNG(12)

    ctx = test_fixture()
    meta = ctx.meta
    kernel = ctx.kernel
    method = ctx.method
    θ_val = ctx.θ_val
    Xu = ctx.Xu
    q_x = ctx.q_x
    sample_x = rand(rng, q_x, 10000)
    μ_v, Σ_v = mean_cov_vector_matrix(ctx.q_v)
    Rv = Σ_v + μ_v * μ_v'
    Wg_bar = mean(ctx.q_Wg)

    Ex = getEx(meta)
    Dxθ = getDxθ(meta)
    Fxθ = getFxθ(meta)
    Cxθ_Xu = getCxθ_Xu(meta)
    mf = getMeanFn(meta)
    Ku_mxu = meta.KuuF \ apply_mean_fn.(meta.Xu, mf)


    # ground truth univariate functions
    Ψx_func = (x) -> apply_mean_fn(x, mf)
    Ψxx_func = (x) -> apply_mean_fn(x, mf) * apply_mean_fn(x, mf)
    Ψ0_func = (x) -> kernelmatrix(kernel(θ_val),[x])[]
    Ψ1_func = (x) -> kernelmatrix(kernel(θ_val), [x], Xu)
    Ψ2_func = (x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu)
    Ψ3_func = (x) -> apply_mean_fn(x, mf) * kernelmatrix(kernel(θ_val), [x], Xu)

    # ground truth univariate expectations
    Ψx_gt = mean(Ψx_func.(sample_x))
    Ψxx_gt = mean(Ψxx_func.(sample_x))
    Ψ0_gt = mean(Ψ0_func.(sample_x))
    Ψ1_gt = mean(Ψ1_func.(sample_x))
    Ψ2_gt = mean(Ψ2_func.(sample_x))
    Ψ3_gt = mean(Ψ3_func.(sample_x))

    # univariate approx expectations
    Ψx_approx = approximate_kernel_expectation(method, (x) -> [apply_mean_fn(x, mf)], q_x)[]
    Ψxx_approx = approximate_kernel_expectation(method, (x) -> [apply_mean_fn(x, mf) * apply_mean_fn(x, mf)], q_x)[]
    Ψ0_approx = approximate_kernel_expectation(method, (x) -> kernelmatrix(kernel(θ_val), [x], [x]), q_x)[]
    Ψ1_approx = approximate_kernel_expectation(method, (x) -> kernelmatrix(kernel(θ_val), [x], Xu), q_x)
    Ψ2_approx = approximate_kernel_expectation(method, (x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu), q_x)
    Ψ3_approx = approximate_kernel_expectation(method, (x) -> apply_mean_fn(x, mf) * kernelmatrix(kernel(θ_val), [x], Xu), q_x)

    # ground truth univariate grad functions
    Ωx_func = (x) -> Ex(x)
    Ω0_func = (x) -> Dxθ(x, θ_val)
    Ω1_func = (x) -> Cxθ_Xu(x, θ_val, Xu)
    Ω2_func = (x) -> transpose(Cxθ_Xu(x, θ_val, Xu)) * Cxθ_Xu(x, θ_val, Xu)
    Ω3_func = (x) -> transpose(Cxθ_Xu(x, θ_val, Xu)) * Wg_bar * Cxθ_Xu(x, θ_val, Xu)
    Ω4_func = (x) -> transpose(Ex(x)) * Wg_bar * Cxθ_Xu(x, θ_val, Xu)
    Ω5_func = (x) -> Ex(x) * transpose(Ex(x))
    Ω6_func = (x) -> Ex(x) * transpose(μ_v) * transpose(Cxθ_Xu(x, θ_val, Xu))
    Ω7_func = (x) -> Ex(x) * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ_val, Xu))
    Ω8_func = (x) -> Cxθ_Xu(x, θ_val, Xu) * μ_v * transpose(Ex(x))
    Ω9_func = (x) -> Cxθ_Xu(x, θ_val, Xu) * Rv * transpose(Cxθ_Xu(x, θ_val, Xu))
    Ω10_func = (x) -> Cxθ_Xu(x, θ_val, Xu) * μ_v * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ_val, Xu))
    Ω11_func = (x) -> Cxθ_Xu(x, θ_val, Xu) * Ku_mxu * transpose(Ex(x))
    Ω12_func = (x) -> Cxθ_Xu(x, θ_val, Xu) * Ku_mxu * transpose(μ_v) * transpose(Cxθ_Xu(x, θ_val, Xu))
    Ω13_func = (x) -> Cxθ_Xu(x, θ_val, Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ_val, Xu))

    # ground truth univariate grad expectations
    Ωx_gt = mean(Ωx_func.(sample_x))
    Ω0_gt = mean(Ω0_func.(sample_x))
    Ω1_gt = mean(Ω1_func.(sample_x))
    Ω2_gt = mean(Ω2_func.(sample_x))
    Ω3_gt = mean(Ω3_func.(sample_x))
    Ω4_gt = mean(Ω4_func.(sample_x))
    Ω5_gt = mean(Ω5_func.(sample_x))
    Ω6_gt = mean(Ω6_func.(sample_x))
    Ω7_gt = mean(Ω7_func.(sample_x))
    Ω8_gt = mean(Ω8_func.(sample_x))
    Ω9_gt = mean(Ω9_func.(sample_x))
    Ω10_gt = mean(Ω10_func.(sample_x))
    Ω11_gt = mean(Ω11_func.(sample_x))
    Ω12_gt = mean(Ω12_func.(sample_x))
    Ω13_gt = mean(Ω13_func.(sample_x))

    # univariate_grad approx expectations
    Ωx_approx = approximate_kernel_expectation(meta.method, (x) -> Ex(x), q_x)
    Ω0_approx = approximate_kernel_expectation(meta.method, (x) -> Dxθ(x, θ_val), q_x)
    Ω1_approx = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ_val, meta.Xu), q_x)
    Ω2_approx = approximate_kernel_expectation(meta.method, (x) -> transpose(Cxθ_Xu(x, θ_val, meta.Xu)) * Cxθ_Xu(x, θ_val, meta.Xu), q_x)
    Ω3_approx = approximate_kernel_expectation(meta.method, (x) -> transpose(Cxθ_Xu(x, θ_val, meta.Xu)) * Wg_bar * Cxθ_Xu(x, θ_val, meta.Xu), q_x)
    Ω4_approx = approximate_kernel_expectation(meta.method, (x) -> transpose(Ex(x)) * Wg_bar * Cxθ_Xu(x, θ_val, meta.Xu), q_x)
    Ω5_approx = approximate_kernel_expectation(meta.method, (x) -> Ex(x) * transpose(Ex(x)), q_x)
    Ω6_approx = approximate_kernel_expectation(meta.method, (x) -> Ex(x) * transpose(μ_v) * transpose(Cxθ_Xu(x, θ_val, meta.Xu)), q_x)
    Ω7_approx = approximate_kernel_expectation(meta.method, (x) -> Ex(x) * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ_val, meta.Xu)), q_x)
    Ω8_approx = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ_val, meta.Xu) * μ_v * transpose(Ex(x)), q_x)
    Ω9_approx = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ_val, meta.Xu) * Rv * transpose(Cxθ_Xu(x, θ_val, meta.Xu)), q_x)
    Ω10_approx = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ_val, meta.Xu) * μ_v * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ_val, meta.Xu)), q_x)
    Ω11_approx = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ_val, meta.Xu) * Ku_mxu * transpose(Ex(x)), q_x)
    Ω12_approx = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ_val, meta.Xu) * Ku_mxu * transpose(μ_v) * transpose(Cxθ_Xu(x, θ_val, meta.Xu)), q_x)
    Ω13_approx = approximate_kernel_expectation(meta.method, (x) -> Cxθ_Xu(x, θ_val, meta.Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Cxθ_Xu(x, θ_val, meta.Xu)), q_x)
    
    # univariate tests
    @test isapprox(Ψx_gt, Ψx_approx ;atol = 5e-3)
    @test isapprox(Ψxx_gt, Ψxx_approx ;atol = 5e-3)
    @test isapprox(Ψ0_gt, Ψ0_approx ;atol = 5e-3)
    @test isapprox(Ψ1_gt, Ψ1_approx ;atol = 5e-3)
    @test isapprox(Ψ2_gt, Ψ2_approx ;atol = 1e-2)
    @test isapprox(Ψ3_gt, Ψ3_approx ;atol = 5e-3)

    # univariate grad tests
    @test isapprox(Ωx_gt, Ωx_approx ;atol = 5e-3)
    @test isapprox(Ω0_gt, Ω0_approx ;atol = 5e-3)
    @test isapprox(Ω1_gt, Ω1_approx ;atol = 5e-3)
    @test isapprox(Ω2_gt, Ω2_approx ;atol = 5e-3)
    @test isapprox(Ω3_gt, Ω3_approx ;atol = 1e-2)
    @test isapprox(Ω4_gt, Ω4_approx ;atol = 1e-2)
    @test isapprox(Ω5_gt, Ω5_approx ;atol = 2e-2)
    @test isapprox(Ω6_gt, Ω6_approx ;atol = 5e-3)
    @test isapprox(Ω7_gt, Ω7_approx ;atol = 2e-2)
    @test isapprox(Ω8_gt, Ω8_approx ;atol = 5e-3)
    @test isapprox(Ω9_gt, Ω9_approx ;atol = 5e-3)
    @test isapprox(Ω10_gt, Ω10_approx ;atol = 5e-3)
    @test isapprox(Ω11_gt, Ω11_approx ;atol = 2e-2)
    @test isapprox(Ω12_gt, Ω12_approx ;atol = 5e-3)
    @test isapprox(Ω13_gt, Ω13_approx ;atol = 2e-2)
end