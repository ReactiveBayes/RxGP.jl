@testitem "node_rule/univariate_batch/Test out rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    import RxGP: approximate_kernel_expectation, approximate_kernel_expectation!

    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D = 1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(StableRNG(42), Nu) |> (x) -> sin.(x), diageye(Nu))
    q_θ = PointMass(θ_val)
    μ_v = mean(q_v)
    mf = getMeanFn(Unimeta)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu

    # Test with 3 PointMass inputs
    x_vals = [0.5, 1.5, 2.5]
    q_in_batch = PointMass(x_vals)

    ν_out_batch = @call_rule UniSGP_Batch(:out, Marginalisation) (q_in = q_in_batch, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_out_batch) <: MultivariateGaussianDistributionsFamily
    @test length(mean(ν_out_batch)) == 3

    # Compare against per-point results
    for (j, xv) in enumerate(x_vals)
        ν_single = @call_rule UniSGP(:out, Marginalisation) (q_in = PointMass(xv), q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
        @test isapprox(mean(ν_out_batch)[j], mean(ν_single); atol=1e-10)
    end
    @test isapprox(var(ν_out_batch), fill(inv(mean(q_w)), 3); atol=1e-10)
end

@testitem "node_rule/univariate_batch/Test v rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    import RxGP: approximate_kernel_expectation

    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D = 1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_w = GammaShapeRate(1,1)
    q_θ = PointMass(θ_val)
    w_bar = mean(q_w)
    mf = getMeanFn(Unimeta)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu

    # Batch: 3 PointMass points with PointMass outputs
    x_vals = [0.5, 1.5, 2.5]
    y_vals = [1.0, 2.0, 3.0]
    q_in_batch = PointMass(x_vals)
    q_out_batch = PointMass(y_vals)

    ν_v_batch = @call_rule UniSGP_Batch(:v, Marginalisation) (q_out = q_out_batch, q_in = q_in_batch, q_w = q_w, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_v_batch) <: MultivariateGaussianDistributionsFamily

    # Compute expected W_v and ξ_v by summing per-point ψ-stats
    W_v_expected = zeros(Nu, Nu)
    rhs_expected = zeros(Nu)
    for (xv, yv) in zip(x_vals, y_vals)
        Ψ1_trans = kernelmatrix(kernel(θ_val), Xu, [xv])
        Ψ2 = kernelmatrix(kernel(θ_val), Xu, [xv]) * kernelmatrix(kernel(θ_val), [xv], Xu) + 1e-8*I
        mx = apply_mean_fn(xv, mf)
        Ψ3 = mx * kernelmatrix(kernel(θ_val), [xv], Xu)
        W_v_expected += w_bar * Ψ2
        rhs_expected += vec(w_bar * (yv * Ψ1_trans - transpose(Ψ3)))
    end
    ξ_v_expected = rhs_expected + W_v_expected * Ku_mxu
    μ_v_expected = W_v_expected \ ξ_v_expected

    @test isapprox(mean(ν_v_batch), μ_v_expected; atol=1e-8)
    @test isapprox(cov(ν_v_batch), inv(W_v_expected); atol=1e-8)
end

@testitem "node_rule/univariate_batch/Test w rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    import RxGP: approximate_kernel_expectation

    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D = 1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_v = MvNormalMeanCovariance(rand(StableRNG(42), Nu) |> (x) -> sin.(x), diageye(Nu))
    q_θ = PointMass(θ_val)
    μ_v = mean(q_v)
    Rv = μ_v * μ_v' + cov(q_v)
    mf = getMeanFn(Unimeta)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu

    x_vals = [0.5, 1.5, 2.5]
    y_vals = [1.0, 2.0, 3.0]
    N_j = length(x_vals)
    q_in_batch = PointMass(x_vals)
    q_out_batch = PointMass(y_vals)

    ν_w_batch = @call_rule UniSGP_Batch(:w, Marginalisation) (q_out = q_out_batch, q_in = q_in_batch, q_v = q_v, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_w_batch) <: GammaDistributionsFamily
    @test shape(ν_w_batch) == 1 + 0.5 * N_j

    # Compute expected rate by summing per-point I1 + I4
    I_sum = 0.0
    for (xv, yv) in zip(x_vals, y_vals)
        Ψx = apply_mean_fn(xv, mf)
        Ψxx = Ψx^2
        Ψ0 = kernelmatrix(kernel(θ_val), [xv], [xv])[1]
        Ψ1_trans = kernelmatrix(kernel(θ_val), Xu, [xv])
        Ψ2 = kernelmatrix(kernel(θ_val), Xu, [xv]) * kernelmatrix(kernel(θ_val), [xv], Xu) + 1e-8*I
        Ψ3 = Ψx * kernelmatrix(kernel(θ_val), [xv], Xu)

        α = Unimeta.KuuF.L \ Ψ1_trans
        I1 = Ψ0 - jdotavx(α, α)
        Ψ2_Ku_mxu = Ψ2 * Ku_mxu
        I4 = yv^2 - 2*yv*(Ψx + jdotavx(Ψ1_trans, μ_v - Ku_mxu)) + Ψxx + tr(Rv * Ψ2) + jdotavx(Ku_mxu, Ψ2_Ku_mxu) + 2*jdotavx(Ψ3, μ_v - Ku_mxu) - 2*jdotavx(μ_v, Ψ2_Ku_mxu)
        I4 = clamp(I4, 1e-12, 1e12)
        I_sum += I1 + I4
    end

    @test isapprox(rate(ν_w_batch), 0.5 * I_sum; atol=1e-8)
end

@testitem "node_rule/univariate_batch/Test in rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test

    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D = 1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_out_batch = PointMass([1.0, 2.0])
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(StableRNG(42), Nu) |> (x) -> sin.(x), diageye(Nu))
    q_θ = PointMass(θ_val)

    ν_in_batch = @call_rule UniSGP_Batch(:in, Marginalisation) (q_out = q_out_batch, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_in_batch) <: ContinuousMultivariateLogPdf

    # Evaluate at a test point and compare with sum of per-point log messages
    x_test_batch = [0.5, 1.5]
    ν_in_1 = @call_rule UniSGP(:in, Marginalisation) (q_out = PointMass(1.0), q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
    ν_in_2 = @call_rule UniSGP(:in, Marginalisation) (q_out = PointMass(2.0), q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)

    lp_batch = logpdf(ν_in_batch, x_test_batch)
    lp_sum = logpdf(ν_in_1, 0.5) + logpdf(ν_in_2, 1.5)
    @test isapprox(lp_batch, lp_sum; atol=1e-10)
end

@testitem "node_rule/univariate_batch/Test theta rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test

    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D = 1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_out_batch = PointMass([1.0, 2.0, 3.0])
    q_in_batch = PointMass([0.5, 1.5, 2.5])
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(StableRNG(42), Nu) |> (x) -> sin.(x), diageye(Nu))

    ν_θ_batch = @call_rule UniSGP_Batch(:θ, Marginalisation) (q_out = q_out_batch, q_in = q_in_batch, q_v = q_v, q_w = q_w, meta = Unimeta)
    @test typeof(ν_θ_batch) <: Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}

    # Evaluate at θ_val and compare with sum of per-point θ messages
    ν_θ_1 = @call_rule UniSGP(:θ, Marginalisation) (q_out = PointMass(1.0), q_in = PointMass(0.5), q_v = q_v, q_w = q_w, meta = Unimeta)
    ν_θ_2 = @call_rule UniSGP(:θ, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.5), q_v = q_v, q_w = q_w, meta = Unimeta)
    ν_θ_3 = @call_rule UniSGP(:θ, Marginalisation) (q_out = PointMass(3.0), q_in = PointMass(2.5), q_v = q_v, q_w = q_w, meta = Unimeta)

    lp_batch = logpdf(ν_θ_batch, θ_val)
    lp_sum = logpdf(ν_θ_1, θ_val) + logpdf(ν_θ_2, θ_val) + logpdf(ν_θ_3, θ_val)
    @test isapprox(lp_batch, lp_sum; atol=1e-8)
end
