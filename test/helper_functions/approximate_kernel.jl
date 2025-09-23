module TestApproximateKernel
using RxGP, RxInfer, ReactiveMP
using Random, Distributions, StableRNGs
using KernelFunctions, LinearAlgebra
using Test


@testset "ApproximateKernel" begin
    method = ghcubature(21)
    rng = StableRNG(12)
    q_x = Normal(0,1)
    θ_val = [1.,1.]
    Xu = collect(1:10) #inducing points for univariate case
    kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])

    sample_x = rand(rng,q_x,10000)
    Ψ2_func = (x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu)
    Ψ0_func = (x) -> getindex(kernelmatrix(kernel(θ_val),[x]),1)
    A_xθ = (x,θ) -> kernelmatrix(kernel(θ),[x],[x]) .- kernelmatrix(kernel(θ),[x],Xu) * inv(kernelmatrix(kernel(θ),Xu)) * kernelmatrix(kernel(θ),Xu,[x])
    B_xθ = (x,θ) -> kernelmatrix(kernel(θ), [x], Xu)
    A_x = (x) -> A_xθ(x,θ_val)
    B_x = (x) -> B_xθ(x,θ_val)

    Ψ0_gt = mean(Ψ0_func.(sample_x))
    Ψ1_gt = mean(B_x.(sample_x))
    Ψ2_gt = mean(Ψ2_func.(sample_x)) + 1e-7*I

    @testset "approximate_kernel_expectation" begin
        #make sure we add new methods for approximate_kernel_expectation properly
        #make sure the function works as expected
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

    @testset "Psi approximation" begin
        Ψ0_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), [x], [x]),q_x)[]
        Ψ1_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), [x], Xu),q_x)
        Ψ2_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu), q_x) + 1e-7*I

        @test isapprox(Ψ0_gt, Ψ0_approx ;atol = 1e-4)
        @test isapprox(Ψ1_gt, Ψ1_approx ;atol = 0.05)
        @test isapprox(Ψ2_gt, Ψ2_approx ;atol = 0.05)
    end
end

end