@testitem "node_rule/univariate_grad_batch/Test out rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    import RxGP: approximate_kernel_expectation

    method = ghcubature(21)
    Nu = 5
    D = 2
    Xu = [randn(StableRNG(42+i), D) for i in 1:Nu]
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_v = MvNormalMeanCovariance(rand(StableRNG(42), Nu) |> (x) -> sin.(x), diageye(Nu))
    q_Wg = PointMass(Matrix(1.0I, D, D))
    q_θ = PointMass(θ_val)
    μ_v = mean(q_v)
    Ex = getEx(Unimeta)
    Cxθ_Xu = getCxθ_Xu(Unimeta)
    mf = getMeanFn(Unimeta)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu

    # Batch: 3 input points, each D-dimensional
    N_j = 3
    x_points = [randn(StableRNG(100+j), D) for j in 1:N_j]
    x_flat = vcat(x_points...)
    q_in_batch = PointMass(x_flat)

    ν_out_batch = @call_rule UniSGP_Grad_Batch(:out, Marginalisation) (q_in = q_in_batch, q_v = q_v, q_Wg = q_Wg, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_out_batch) <: MultivariateGaussianDistributionsFamily
    @test length(mean(ν_out_batch)) == D * N_j

    # Compare per-point
    for j in 1:N_j
        ν_single = @call_rule UniSGP_Grad(:out, Marginalisation) (q_in = PointMass(x_points[j]), q_v = q_v, q_Wg = q_Wg, q_θ = q_θ, meta = Unimeta)
        idx = (j-1)*D+1 : j*D
        @test isapprox(mean(ν_out_batch)[idx], mean(ν_single); atol=1e-10)
    end
end

@testitem "node_rule/univariate_grad_batch/Test v rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    import RxGP: approximate_kernel_expectation

    method = ghcubature(21)
    Nu = 5
    D = 2
    Xu = [randn(StableRNG(42+i), D) for i in 1:Nu]
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_Wg = PointMass(Matrix(1.0I, D, D))
    Wg_bar = mean(q_Wg)
    q_θ = PointMass(θ_val)
    Ex = getEx(Unimeta)
    Cxθ_Xu = getCxθ_Xu(Unimeta)
    mf = getMeanFn(Unimeta)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu

    N_j = 3
    x_points = [randn(StableRNG(100+j), D) for j in 1:N_j]
    ω_points = [randn(StableRNG(200+j), D) for j in 1:N_j]
    x_flat = vcat(x_points...)
    ω_flat = vcat(ω_points...)
    q_in_batch = PointMass(x_flat)
    q_out_batch = PointMass(ω_flat)

    ν_v_batch = @call_rule UniSGP_Grad_Batch(:v, Marginalisation) (q_out = q_out_batch, q_in = q_in_batch, q_Wg = q_Wg, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_v_batch) <: MultivariateGaussianDistributionsFamily

    # Compute expected by summing per-point
    M = Nu
    Ω3_sum = zeros(M, M)
    ξ_accum = zeros(M)
    for j in 1:N_j
        Ω1 = Cxθ_Xu(x_points[j], θ_val, Xu)
        Ω3 = transpose(Ω1) * Wg_bar * Ω1
        Ω4 = transpose(Ex(x_points[j])) * Wg_bar * Ω1
        Ω3_sum += Ω3
        ξ_accum += vec(Ω3 * Ku_mxu + transpose(Ω1) * Wg_bar * ω_points[j] - transpose(Ω4))
    end
    W_v_expected = Symmetric(Ω3_sum + 1e-8I, :U)
    μ_v_expected = W_v_expected \ ξ_accum

    @test isapprox(mean(ν_v_batch), μ_v_expected; atol=1e-8)
    @test isapprox(cov(ν_v_batch), inv(Matrix(W_v_expected)); atol=1e-8)
end

@testitem "node_rule/univariate_grad_batch/Test Wg rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    import RxGP: approximate_kernel_expectation
    import ReactiveMP: WishartFast

    method = ghcubature(21)
    Nu = 5
    D = 2
    Xu = [randn(StableRNG(42+i), D) for i in 1:Nu]
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_v = MvNormalMeanCovariance(rand(StableRNG(42), Nu) |> (x) -> sin.(x), diageye(Nu))
    q_θ = PointMass(θ_val)
    μ_v, Σ_v = mean_cov(q_v)
    Rv = μ_v * transpose(μ_v) + Σ_v

    N_j = 3
    x_points = [randn(StableRNG(100+j), D) for j in 1:N_j]
    ω_points = [randn(StableRNG(200+j), D) for j in 1:N_j]
    x_flat = vcat(x_points...)
    ω_flat = vcat(ω_points...)
    q_in_batch = PointMass(x_flat)
    q_out_batch = PointMass(ω_flat)

    ν_Wg_batch = @call_rule UniSGP_Grad_Batch(:Wg, Marginalisation) (q_out = q_out_batch, q_in = q_in_batch, q_v = q_v, q_θ = q_θ, meta = Unimeta)

    # Check degrees of freedom
    @test ν_Wg_batch.ν == N_j + D + 1
end

@testitem "node_rule/univariate_grad_batch/Test in rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test

    method = ghcubature(21)
    Nu = 5
    D = 2
    Xu = [randn(StableRNG(42+i), D) for i in 1:Nu]
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_v = MvNormalMeanCovariance(rand(StableRNG(42), Nu) |> (x) -> sin.(x), diageye(Nu))
    q_Wg = PointMass(Matrix(1.0I, D, D))
    q_θ = PointMass(θ_val)

    N_j = 2
    ω_points = [randn(StableRNG(200+j), D) for j in 1:N_j]
    ω_flat = vcat(ω_points...)
    q_out_batch = PointMass(ω_flat)

    ν_in_batch = @call_rule UniSGP_Grad_Batch(:in, Marginalisation) (q_out = q_out_batch, q_v = q_v, q_Wg = q_Wg, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_in_batch) <: ContinuousMultivariateLogPdf

    # Evaluate at a test point and compare with sum of per-point messages
    x_test_points = [randn(StableRNG(300+j), D) for j in 1:N_j]
    x_test_flat = vcat(x_test_points...)

    ν_in_1 = @call_rule UniSGP_Grad(:in, Marginalisation) (q_out = PointMass(ω_points[1]), q_v = q_v, q_Wg = q_Wg, q_θ = q_θ, meta = Unimeta)
    ν_in_2 = @call_rule UniSGP_Grad(:in, Marginalisation) (q_out = PointMass(ω_points[2]), q_v = q_v, q_Wg = q_Wg, q_θ = q_θ, meta = Unimeta)

    lp_batch = logpdf(ν_in_batch, x_test_flat)
    lp_sum = logpdf(ν_in_1, x_test_points[1]) + logpdf(ν_in_2, x_test_points[2])
    @test isapprox(lp_batch, lp_sum; atol=1e-8)
end

@testitem "node_rule/univariate_grad_batch/Test theta rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test

    method = ghcubature(21)
    Nu = 5
    D = 2
    Xu = [randn(StableRNG(42+i), D) for i in 1:Nu]
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)

    q_v = MvNormalMeanCovariance(rand(StableRNG(42), Nu) |> (x) -> sin.(x), diageye(Nu))
    q_Wg = PointMass(Matrix(1.0I, D, D))
    q_θ_pm = PointMass(θ_val)

    N_j = 2
    x_points = [randn(StableRNG(100+j), D) for j in 1:N_j]
    ω_points = [randn(StableRNG(200+j), D) for j in 1:N_j]
    x_flat = vcat(x_points...)
    ω_flat = vcat(ω_points...)
    q_in_batch = PointMass(x_flat)
    q_out_batch = PointMass(ω_flat)

    ν_θ_batch = @call_rule UniSGP_Grad_Batch(:θ, Marginalisation) (q_out = q_out_batch, q_in = q_in_batch, q_v = q_v, q_Wg = q_Wg, meta = Unimeta)
    @test typeof(ν_θ_batch) <: Union{ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf}

    # Compare with sum of per-point θ messages
    ν_θ_1 = @call_rule UniSGP_Grad(:θ, Marginalisation) (q_out = PointMass(ω_points[1]), q_in = PointMass(x_points[1]), q_v = q_v, q_Wg = q_Wg, meta = Unimeta)
    ν_θ_2 = @call_rule UniSGP_Grad(:θ, Marginalisation) (q_out = PointMass(ω_points[2]), q_in = PointMass(x_points[2]), q_v = q_v, q_Wg = q_Wg, meta = Unimeta)

    lp_batch = logpdf(ν_θ_batch, θ_val)
    lp_sum = logpdf(ν_θ_1, θ_val) + logpdf(ν_θ_2, θ_val)
    @test isapprox(lp_batch, lp_sum; atol=1e-8)
end
