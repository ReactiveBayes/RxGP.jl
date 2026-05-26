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

    const grad_default_method = ReactiveMP.ghcubature(21)

    test_mean_fn(x) = begin
        sum(x)
    end

    function test_fixture(;D=1)
        rng = StableRNG(12)
        Nu = 5
        if D == 1
            q_x = NormalMeanVariance(rand(rng), rand(rng) + 0.1)
            q_out = NormalMeanVariance(rand(rng), rand(rng) + 0.1)
            q_Wg = PointMass(rand(rng) + 0.1)
            Xu = [rand(rng) for _ in 1:Nu]
        elseif D > 1
            q_x = MvNormalMeanCovariance(randn(rng, D), randn(rng, D, D) |> x -> x * x' + 0.1I)
            q_out = NormalMeanVariance(rand(rng), rand(rng) + 0.1)
            q_Wg = PointMass(randn(rng, D, D) |> x -> x * x' + 0.1I)
            Xu = [randn(rng, D) for _ in 1:Nu]
        end
        q_v = MvNormalMeanCovariance(randn(rng, Nu), randn(rng, Nu, Nu) |> x -> x * x' + 0.1I)
        kernel_spec = :SEn
        kernel, θ_val, _ = get_simple_kernel_and_params(D; kernel_spec=kernel_spec)
        meta = get_UniSGPMeta(D; method=grad_default_method, mean_fn=test_mean_fn, kernel=kernel, kernel_spec=kernel_spec, mode=:AN, independent_SE_lengthscales=true, Xu=Xu, operator=:grad, θ=θ_val)
        return (; method=grad_default_method, D, Xu, Nu=length(Xu), kernel, θ_val, meta, q_x, q_out, q_v, q_Wg, q_θ=PointMass(θ_val), rng)
    end
end

@testitem "helper_functions/approximate_kernel/approximate_kernel_expectation" setup=[setup_snippet] begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    import RxGP: approximate_kernel_expectation, approximate_kernel_expectation!
    
    ctx = test_fixture(;D=1)
    method = ctx.method
    q_x = ctx.q_x

    # make sure we add new methods for approximate_kernel_expectation properly
    # make sure the function works as expected
    foo1(x) = 2*x
    @test approximate_kernel_expectation(method,foo1,q_x) == approximate_kernel_expectation(method, foo1, mean(q_x),var(q_x))
    @test isapprox(approximate_kernel_expectation(method,foo1,q_x),2*mean(q_x); atol=1e-9)
    @test approximate_kernel_expectation(GenUT(),foo1,q_x) ≈ 2*mean(q_x)

    foo1_2d(x) = [2*x, x]
    gbar = [1.5,0.5]
    @test approximate_kernel_expectation!(gbar, method, foo1_2d, mean(q_x), var(q_x)) == gbar
    @test approximate_kernel_expectation!(gbar, method, foo1_2d, q_x) == approximate_kernel_expectation!(gbar, method, foo1_2d, mean(q_x), var(q_x))
    @test approximate_kernel_expectation(method,foo1_2d,q_x) == approximate_kernel_expectation(method, foo1_2d, mean(q_x),var(q_x))
    @test isapprox(approximate_kernel_expectation(method,foo1_2d,q_x),[2*mean(q_x), mean(q_x)]; atol=1e-9)
    @test approximate_kernel_expectation(GenUT(),foo1_2d,q_x) ≈ [2*mean(q_x), mean(q_x)]

    
    ctx = test_fixture(;D=3)
    method = ctx.method
    q_x = ctx.q_x
    μ_x = mean(q_x)

    # make sure we add new methods for approximate_kernel_expectation properly
    # make sure the function works as expected
    foo2(x) = 2*sum(x)
    @test approximate_kernel_expectation(method,foo2,q_x) == approximate_kernel_expectation(method, foo2, mean(q_x),cov(q_x))
    @test isapprox(approximate_kernel_expectation(method,foo2,q_x),2*sum(μ_x); atol=1e-9)

    foo2_2d(x) = [2*sum(x), sum(x)]
    gbar = [1.5,0.5]
    @test approximate_kernel_expectation!(gbar, method, foo2_2d, mean(q_x), cov(q_x)) == gbar
    @test approximate_kernel_expectation!(gbar, method, foo2_2d, q_x) == approximate_kernel_expectation!(gbar, method, foo2_2d, mean(q_x), cov(q_x))
    @test approximate_kernel_expectation(method,foo2_2d,q_x) == approximate_kernel_expectation(method, foo2_2d, mean(q_x),cov(q_x))
    @test isapprox(approximate_kernel_expectation(method,foo2_2d,q_x),[2*sum(μ_x), sum(μ_x)]; atol=1e-9)
end

@testitem "helper_functions/approximate_kernel/GP quantities approximation" setup=[setup_snippet] begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    import RxGP: approximate_kernel_expectation, approximate_kernel_expectation!

    ctx = test_fixture(;D=3)
    meta = ctx.meta
    kernel = ctx.kernel
    method = ctx.method
    θ_val = ctx.θ_val
    Xu = ctx.Xu
    q_x = ctx.q_x
    sample_x = rand(ctx.rng, q_x, 20000)
    if ctx.D > 1
        sample_x = [col for col in eachcol(sample_x)]
    end
    μ_v, Σ_v = mean_cov_vector_matrix(ctx.q_v)
    Rv = Σ_v + μ_v * μ_v'
    Wg_bar = mean(ctx.q_Wg)
    println("sizeWg_bar: ", size(Wg_bar))
    Lm_fn = getLm_fn(meta)
    Kxx_fn = getKxx_fn(meta)
    Kxu_fn = getKxu_fn(meta)
    mf = getMeanFn(meta)
    Ku_mxu = meta.KuuF \ apply_mean_fn.(meta.Xu, mf)

    # ground truth univariate functions
    Ψx_func = (x) -> apply_mean_fn(x, mf)
    Ψxx_func = (x) -> apply_mean_fn(x, mf) * apply_mean_fn(x, mf)
    Ψ0_func = (x) -> kernelmatrix(kernel(θ_val),[x], [x])[1]
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
    Ωx_func = (x) -> Lm_fn(x)
    Ω0_func = (x) -> Kxx_fn(x, θ_val)
    Ω1_func = (x) -> Kxu_fn(x, θ_val, Xu)
    Ω2_func = (x) -> transpose(Kxu_fn(x, θ_val, Xu)) * Kxu_fn(x, θ_val, Xu)
    Ω3_func = (x) -> transpose(Kxu_fn(x, θ_val, Xu)) * Wg_bar * Kxu_fn(x, θ_val, Xu)
    Ω4_func = (x) -> transpose(Lm_fn(x)) * Wg_bar * Kxu_fn(x, θ_val, Xu)
    Ω5_func = (x) -> Lm_fn(x) * transpose(Lm_fn(x))
    Ω6_func = (x) -> Lm_fn(x) * transpose(μ_v) * transpose(Kxu_fn(x, θ_val, Xu))
    Ω7_func = (x) -> Lm_fn(x) * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ_val, Xu))
    Ω8_func = (x) -> Kxu_fn(x, θ_val, Xu) * μ_v * transpose(Lm_fn(x))
    Ω9_func = (x) -> Kxu_fn(x, θ_val, Xu) * Rv * transpose(Kxu_fn(x, θ_val, Xu))
    Ω10_func = (x) -> Kxu_fn(x, θ_val, Xu) * μ_v * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ_val, Xu))
    Ω11_func = (x) -> Kxu_fn(x, θ_val, Xu) * Ku_mxu * transpose(Lm_fn(x))
    Ω12_func = (x) -> Kxu_fn(x, θ_val, Xu) * Ku_mxu * transpose(μ_v) * transpose(Kxu_fn(x, θ_val, Xu))
    Ω13_func = (x) -> Kxu_fn(x, θ_val, Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ_val, Xu))

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

    # univariate_dID approx expectations
    Ωx_approx = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x), q_x)
    Ω0_approx = approximate_kernel_expectation(meta.method, (x) -> Kxx_fn(x, θ_val), q_x)
    Ω1_approx = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ_val, meta.Xu), q_x)
    Ω2_approx = approximate_kernel_expectation(meta.method, (x) -> transpose(Kxu_fn(x, θ_val, meta.Xu)) * Kxu_fn(x, θ_val, meta.Xu), q_x)
    Ω3_approx = approximate_kernel_expectation(meta.method, (x) -> transpose(Kxu_fn(x, θ_val, meta.Xu)) * Wg_bar * Kxu_fn(x, θ_val, meta.Xu), q_x)
    Ω4_approx = approximate_kernel_expectation(meta.method, (x) -> transpose(Lm_fn(x)) * Wg_bar * Kxu_fn(x, θ_val, meta.Xu), q_x)
    Ω5_approx = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x) * transpose(Lm_fn(x)), q_x)
    Ω6_approx = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x) * transpose(μ_v) * transpose(Kxu_fn(x, θ_val, meta.Xu)), q_x)
    Ω7_approx = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x) * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ_val, meta.Xu)), q_x)
    Ω8_approx = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ_val, meta.Xu) * μ_v * transpose(Lm_fn(x)), q_x)
    Ω9_approx = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ_val, meta.Xu) * Rv * transpose(Kxu_fn(x, θ_val, meta.Xu)), q_x)
    Ω10_approx = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ_val, meta.Xu) * μ_v * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ_val, meta.Xu)), q_x)
    Ω11_approx = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ_val, meta.Xu) * Ku_mxu * transpose(Lm_fn(x)), q_x)
    Ω12_approx = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ_val, meta.Xu) * Ku_mxu * transpose(μ_v) * transpose(Kxu_fn(x, θ_val, meta.Xu)), q_x)
    Ω13_approx = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ_val, meta.Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ_val, meta.Xu)), q_x)
    
    # univariate tests
    @test isapprox(Ψx_gt, Ψx_approx ;atol = 5e-2)
    @test isapprox(Ψxx_gt, Ψxx_approx ;atol = 5e-2)
    @test isapprox(Ψ0_gt, Ψ0_approx ;atol = 5e-3)
    @test isapprox(Ψ1_gt, Ψ1_approx ;atol = 5e-3)
    @test isapprox(Ψ2_gt, Ψ2_approx ;atol = 1e-2)
    @test isapprox(Ψ3_gt, Ψ3_approx ;atol = 5e-2)

    # univariate grad tests
    @test isapprox(Ωx_gt, Ωx_approx ;atol = 1e-2)
    @test isapprox(Ω0_gt, Ω0_approx ;atol = 5e-3)
    @test isapprox(Ω1_gt, Ω1_approx ;atol = 1e-2)
    @test isapprox(Ω2_gt, Ω2_approx ;atol = 5e-2)
    @test isapprox(Ω3_gt, Ω3_approx ;atol = 5e-2)
    @test isapprox(Ω4_gt, Ω4_approx ;atol = 5e-2)
    @test isapprox(Ω5_gt, Ω5_approx ;atol = 5e-2)
    @test isapprox(Ω6_gt, Ω6_approx ;atol = 5e-2)
    @test isapprox(Ω7_gt, Ω7_approx ;atol = 5e-2)
    @test isapprox(Ω8_gt, Ω8_approx ;atol = 5e-2)
    @test isapprox(Ω9_gt, Ω9_approx ;atol = 5e-2)
    @test isapprox(Ω10_gt, Ω10_approx ;atol = 5e-2)
    @test isapprox(Ω11_gt, Ω11_approx ;atol = 5e-2)
    @test isapprox(Ω12_gt, Ω12_approx ;atol = 5e-2)
    @test isapprox(Ω13_gt, Ω13_approx ;atol = 2e-2)
end