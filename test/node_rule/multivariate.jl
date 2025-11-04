using RxGP, RxInfer, ReactiveMP
using Random, Distributions, StableRNGs
using KernelFunctions, LinearAlgebra
using Optim, Zygote
using Test
using TestItemRunner

@testitem "node_rule/multivariate/Test GPmeta" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = srcubature()
    θ_val = [1.,1.]
    Nu_2d = 25
    Xu_2d = [[i,j] for i=1:5, j=1:5] |> (x) -> reshape(x,Nu_2d)
    kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
    C = [1. 0.;0. 1.]
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I)
    gpcache = GPCache()
    Ψ0 = [1.0;;]
    Ψ1_trans_2d = kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]])
    Ψ2_2d = kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d) + 1e-12*I
    Multimeta = MultiSGPMeta(method, Xu_2d,Ψ0,Ψ1_trans_2d,Ψ2_2d,Kuu_inverse, kernel, gpcache)
    @test getInducingInput(Multimeta) == Xu_2d
    @test getKernel(Multimeta) == kernel
    @test typeof(getKernel(Multimeta)) <: Function
    @test getmethod(Multimeta) == method
    @test getGPCache(Multimeta) == gpcache
    @test getKuuInverse(Multimeta) == Kuu_inverse
    @test getΨ0(Multimeta) == Ψ0 
    @test getΨ1_trans(Multimeta) == Ψ1_trans_2d
    @test getΨ2(Multimeta) == Ψ2_2d
end

@testitem "node_rule/multivariate/Test out rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = srcubature()
    θ_val = [1.,1.]
    Nu_2d = 25
    Xu_2d = [[i,j] for i=1:5, j=1:5] |> (x) -> reshape(x,Nu_2d)
    kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
    C = [1. 0.;0. 1.]
    gpcache = GPCache()
    Ψ0 = [1.0;;]
    Ψ1_trans_2d = kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]])
    Ψ2_2d = kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d) + 1e-12*I
    Multimeta = MultiSGPMeta(method, Xu_2d,Ψ0,Ψ1_trans_2d,Ψ2_2d,cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I), kernel, gpcache)
    q_out = MvNormal([0.5, 1.4], diageye(2))
    q_in =  MvNormal([1.0, 2.7], diageye(2))
    q_v = MvNormalMeanCovariance(rand(2*Nu_2d) |> (x) -> sin.(x), diageye(2*Nu_2d))
    q_w = Wishart(10, 50*diageye(2))
    q_θ = PointMass(θ_val)
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W_mean = mean(q_w)
    Ψ1_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), [x], Xu_2d),q_in)
    mean_out_gt = kron(C,Ψ1_approx) * μ_v
    cov_out_gt = inv(W_mean)
    ν_out = @call_rule MultiSGP(:out, Marginalisation) (q_in = q_in, q_v = q_v, q_w = PointMass(W_mean), q_θ = q_θ, meta = Multimeta)
    @test typeof(ν_out) <: MultivariateGaussianDistributionsFamily
    @test mean(ν_out) ≈ mean_out_gt
    @test cov(ν_out) == cov_out_gt

    ν_out = @call_rule MultiSGP(:out, Marginalisation) (q_in = q_in, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Multimeta)
    @test typeof(ν_out) <: MultivariateGaussianDistributionsFamily
    @test mean(ν_out) ≈ mean_out_gt
    @test cov(ν_out) == cov_out_gt
end

@testitem "node_rule/multivariate/Test in rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Optim, Zygote, Test
    method = srcubature()
    θ_val = [1.,1.]
    Nu_2d = 25
    Xu_2d = [[i,j] for i=1:5, j=1:5] |> (x) -> reshape(x,Nu_2d)
    kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
    C = [1. 0.;0. 1.]
    q_out = MvNormal([0.5, 1.4], diageye(2))
    q_in =  MvNormal([1.0, 2.7], diageye(2))
    q_v = MvNormalMeanCovariance(rand(2*Nu_2d) |> (x) -> sin.(x), diageye(2*Nu_2d))
    q_w = Wishart(10, 50*diageye(2))
    q_θ = PointMass(θ_val)
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    R_v = μ_v * μ_v' + Σ_v
    W_mean = mean(q_w)
    A_2d_xθ = (x,θ) -> kernelmatrix(kernel(θ),[x],[x]) - kernelmatrix(kernel(θ),[x],Xu_2d) * inv(kernelmatrix(kernel(θ),Xu_2d) + 1e-12*I) * kernelmatrix(kernel(θ),Xu_2d,[x])
    B_2d_xθ = (x,θ) -> kernelmatrix(kernel(θ), [x], Xu_2d)
    A_x = (x) -> A_2d_xθ(x,θ_val)
    B_x = (x) -> B_2d_xθ(x,θ_val)
    Multimeta = MultiSGPMeta(method, Xu_2d,[1.0;;],kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]),kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d) + 1e-12*I,cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I), kernel, GPCache())
    gt_logbackwardmess_in = (x) -> -0.5 * tr(W_mean * kron(C,A_x(x))) + μ_y' * W_mean * kron(C,B_x(x)) * μ_v - 0.5*tr(R_v * kron(C,B_x(x))' * W_mean * kron(C,B_x(x)))
    ν_in = @call_rule MultiSGP(:in, Marginalisation) (q_out = q_out, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Multimeta)
    @test typeof(ν_in) <: ContinuousMultivariateLogPdf
    @test logpdf(ν_in, [1.0, 1.5]) ≈ gt_logbackwardmess_in([1.0, 1.5])
    @test logpdf(ν_in, [-1.5, 2.0]) ≈ gt_logbackwardmess_in([-1.5, 2.0])

    q_out_pm = PointMass([1.5,2.0])
    μ_y_pm = [1.5,2.0]
    q_w_pm = PointMass(W_mean)
    gt_logbackwardmess_in = (x) -> -0.5 * tr(W_mean * kron(C,A_x(x))) + μ_y_pm' * W_mean * kron(C,B_x(x)) * μ_v - 0.5*tr(R_v * kron(C,B_x(x))' * W_mean * kron(C,B_x(x)))
    gt_neg_logbackwardmess_in = (x) -> - gt_logbackwardmess_in(x)
    res = optimize(gt_neg_logbackwardmess_in,mean(q_in),LBFGS(),Optim.Options(iterations=20))
    m_z = res.minimizer
    W_z = Zygote.hessian(gt_neg_logbackwardmess_in, m_z)
    ν_in = @call_rule MultiSGP(:in, Marginalisation) (q_out = q_out_pm, q_in = q_in, q_v = q_v, q_w = q_w_pm, q_θ = q_θ, meta = Multimeta)
    @test typeof(ν_in) <: MultivariateGaussianDistributionsFamily
    @test isapprox(mean(ν_in), m_z;atol=0.01)
    @test isapprox(cov(ν_in), inv(W_z);atol=0.01)
end

@testitem "node_rule/multivariate/Test v rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = srcubature()
    θ_val = [1.,1.]
    Nu_2d = 25
    Xu_2d = [[i,j] for i=1:5, j=1:5] |> (x) -> reshape(x,Nu_2d)
    kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
    C = [1. 0.;0. 1.]
    q_out = MvNormal([0.5, 1.4], diageye(2))
    q_in =  MvNormal([1.0, 2.7], diageye(2))
    q_v = MvNormalMeanCovariance(rand(2*Nu_2d) |> (x) -> sin.(x), diageye(2*Nu_2d))
    q_w = Wishart(10, 50*diageye(2))
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    W_mean = mean(q_w)
    Ψ1_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), [x], Xu_2d),q_in)
    Ψ2_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), Xu_2d, [x]) * kernelmatrix(kernel(θ_val), [x], Xu_2d), q_in) + 1e-12*I
    Ψ3 = kron(W_mean, Ψ2_approx)
    Ψ1_tilde = kron(C, Ψ1_approx)
    gt_mean_v = cholinv(Ψ3) * Ψ1_tilde' * W_mean * μ_y 
    gt_cov_v = cholinv(Ψ3)
    ν_v = @call_rule MultiSGP(:v, Marginalisation) (q_out = q_out, q_in = q_in, q_w = q_w, q_θ = PointMass(θ_val), meta = MultiSGPMeta(method, Xu_2d,[1.0;;],kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]),kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d) + 1e-12*I,cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I), kernel, GPCache()))
    @test typeof(ν_v) <: MultivariateGaussianDistributionsFamily
    @test isapprox(mean(ν_v), gt_mean_v; atol = 1e-3)
    @test isapprox(inv(cov(ν_v)) ,inv(gt_cov_v); atol = 1e-1)

    q_out_pm = PointMass([1.5,2.0])
    Ψ3 = kron(W_mean, Ψ2_approx) + 1e-12*I
    Ψ1_tilde = kron(C, Ψ1_approx)
    gt_mean_v = cholinv(Ψ3) * Ψ1_tilde' * W_mean * mean(q_out_pm)
    gt_cov_v = cholinv(Ψ3)
    ν_v = @call_rule MultiSGP(:v, Marginalisation) (q_out = q_out_pm, q_in = q_in, q_w = PointMass(W_mean), q_θ = PointMass(θ_val), meta = MultiSGPMeta(method, Xu_2d,[1.0;;],kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]),kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d) + 1e-12*I,cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I), kernel, GPCache()))
    @test typeof(ν_v) <: MultivariateGaussianDistributionsFamily 
    @test isapprox(mean(ν_v), gt_mean_v;atol=1e-2) 
    @test isapprox(inv(cov(ν_v)), inv(gt_cov_v); atol=1e-1)
end

@testitem "node_rule/multivariate/Test w rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = srcubature()
    θ_val = [1.,1.]
    Nu_2d = 25
    Xu_2d = [[i,j] for i=1:5, j=1:5] |> (x) -> reshape(x,Nu_2d)
    kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
    C = [1. 0.;0. 1.]
    q_out = MvNormal([0.5, 1.4], diageye(2))
    q_in =  MvNormal([1.0, 2.7], diageye(2))
    q_v = MvNormalMeanCovariance(rand(2*Nu_2d) |> (x) -> sin.(x), diageye(2*Nu_2d))
    q_w = Wishart(10, 50*diageye(2))
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    R_v = μ_v * μ_v' + Σ_v
    W_mean = mean(q_w)
    Ψ0_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), [x], [x]),q_in)[]
    Ψ1_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), [x], Xu_2d),q_in)
    Ψ2_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), Xu_2d, [x]) * kernelmatrix(kernel(θ_val), [x], Xu_2d), q_in) + 1e-12*I
    Ψ1_tilde = kron(C, Ψ1_approx)
    Ψ4_approx = approximate_kernel_expectation(srcubature(),(x) -> kron(C,kernelmatrix(kernel(θ_val), [x], Xu_2d)) * R_v * kron(C,kernelmatrix(kernel(θ_val), Xu_2d, [x])), q_in) + 1e-12*I
    I1 = kron(C, Ψ0_approx - tr(cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I) * Ψ2_approx))
    I2 = μ_y * μ_y' + Σ_y - μ_y * μ_v' * Ψ1_tilde' - Ψ1_tilde * μ_v * μ_y' + Ψ4_approx 
    gt_n_w = length(mean(q_out)) + 2
    gt_V_w = cholinv(I1 + I2)
    ν_w = @call_rule MultiSGP(:w, Marginalisation) (q_out = q_out, q_in = q_in, q_v = q_v, q_θ = PointMass(θ_val), meta = MultiSGPMeta(method, Xu_2d,[1.0;;],kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]),kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d) + 1e-12*I,cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I), kernel, GPCache()))
    @test typeof(ν_w) <: WishartDistributionsFamily
    n_w, V_w = params(ν_w)
    @test n_w == gt_n_w 
    @test isapprox(gt_V_w ,V_w; atol=1e-7)
end

@testitem "node_rule/multivariate/Test θ rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = srcubature()
    θ_val = [1.,1.]
    Nu_2d = 25
    Xu_2d = [[i,j] for i=1:5, j=1:5] |> (x) -> reshape(x,Nu_2d)
    kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
    C = [1. 0.;0. 1.]
    q_out = MvNormal([0.5, 1.4], diageye(2))
    q_in =  MvNormal([1.0, 2.7], diageye(2))
    q_v = MvNormalMeanCovariance(rand(2*Nu_2d) |> (x) -> sin.(x), diageye(2*Nu_2d))
    q_w = Wishart(10, 50*diageye(2))
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    R_v = μ_v * μ_v' + Σ_v
    W_mean = mean(q_w)
    Kuu_inverse_θ = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu_2d))
    Ψ0_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[]
    Ψ1_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), [x], Xu_2d),q_in)
    Ψ2_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), Xu_2d, [x]) * kernelmatrix(kernel(θ), [x], Xu_2d), q_in) + 1e-7*I
    I1_θ = (θ) -> kron(C, Ψ0_θ(θ) - tr(Kuu_inverse_θ(θ) * Ψ2_θ(θ)))
    Ψ1_tilde = (θ) -> kron(C, Ψ1_θ(θ))
    Ψ3_θ = (θ) -> kron(W_mean, Ψ2_θ(θ))
    gt_logbackwardmess_θ = (θ) -> -0.5 * tr(W_mean * I1_θ(θ)) + μ_y' * W_mean * Ψ1_tilde(θ) * μ_v - 0.5 * tr(Ψ3_θ(θ) * R_v)
    ν_θ = @call_rule MultiSGP(:θ, Marginalisation) (q_out = q_out, q_in = q_in, q_v = q_v, q_w = q_w, meta = MultiSGPMeta(method, Xu_2d,[1.0;;],kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]),kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d) + 1e-12*I,cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I), kernel, GPCache()))
    @test typeof(ν_θ) <: ContinuousMultivariateLogPdf
    @test logpdf(ν_θ, [1.2, 2.3]) ≈ gt_logbackwardmess_θ([1.2, 2.3])
    @test logpdf(ν_θ, [0.5, 1.4]) ≈ gt_logbackwardmess_θ([0.5, 1.4])

    q_out_pm = PointMass([1.5,2.0])
    μ_y_pm = mean(q_out_pm)
    q_w_pm = PointMass(W_mean)
    Ψ0_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[]
    Ψ1_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), [x], Xu_2d),q_in)
    Ψ2_θ = (θ) -> approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ), Xu_2d, [x]) * kernelmatrix(kernel(θ), [x], Xu_2d), q_in) + 1e-7*I
    I1_θ = (θ) -> kron(C, Ψ0_θ(θ) - tr(Kuu_inverse_θ(θ) * Ψ2_θ(θ)))
    Ψ1_tilde = (θ) -> kron(C, Ψ1_θ(θ))
    Ψ3_θ = (θ) -> kron(W_mean, Ψ2_θ(θ))
    gt_logbackwardmess_θ = (θ) -> -0.5 * tr(W_mean * I1_θ(θ)) + μ_y_pm' * W_mean * Ψ1_tilde(θ) * μ_v - 0.5 * tr(Ψ3_θ(θ) * R_v)
    ν_θ = @call_rule MultiSGP(:θ, Marginalisation) (q_out = q_out_pm, q_in = q_in, q_v = q_v, q_w = q_w_pm, meta = MultiSGPMeta(method, Xu_2d,[1.0;;],kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]),kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d) + 1e-12*I,cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I), kernel, GPCache()))
    @test typeof(ν_θ) <: ContinuousMultivariateLogPdf
    @test logpdf(ν_θ, [1.2, 2.3]) ≈ gt_logbackwardmess_θ([1.2, 2.3])
    @test logpdf(ν_θ, [0.5, 1.4]) ≈ gt_logbackwardmess_θ([0.5, 1.4])
end

@testitem "node_rule/multivariate/Average energy" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = srcubature()
    θ_val = [1.,1.]
    Nu_2d = 25
    Xu_2d = [[i,j] for i=1:5, j=1:5] |> (x) -> reshape(x,Nu_2d)
    kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
    C = [1. 0.;0. 1.]
    q_out = MvNormal([0.5, 1.4], diageye(2))
    q_in =  MvNormal([1.0, 2.7], diageye(2))
    q_v = MvNormalMeanCovariance(rand(2*Nu_2d) |> (x) -> sin.(x), diageye(2*Nu_2d))
    q_w = Wishart(10, 50*diageye(2))
    q_θ = PointMass(θ_val)
    μ_y, Σ_y = mean_cov(q_out)
    μ_v, Σ_v = mean_cov(q_v)
    R_v = μ_v * μ_v' + Σ_v
    W_mean = mean(q_w)
    E_logdet_W = mean(logdet, q_w)
    Kuu_inverse = cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I)
    Ψ0_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), [x], [x]),q_in)[]
    Ψ1_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), [x], Xu_2d),q_in)
    Ψ2_approx = approximate_kernel_expectation(srcubature(),(x) -> kernelmatrix(kernel(θ_val), Xu_2d, [x]) * kernelmatrix(kernel(θ_val), [x], Xu_2d), q_in) + 1e-12*I
    Ψ1_tilde = kron(C, Ψ1_approx)
    Ψ4_approx = approximate_kernel_expectation(srcubature(),(x) -> kron(C,kernelmatrix(kernel(θ_val), [x], Xu_2d)) * R_v * kron(C,kernelmatrix(kernel(θ_val), Xu_2d, [x])), q_in)
    I1 = kron(C, Ψ0_approx - tr(Kuu_inverse * Ψ2_approx))
    I2 = μ_y * μ_y' + Σ_y - μ_y * μ_v' * Ψ1_tilde' - Ψ1_tilde * μ_v * μ_y' + Ψ4_approx 
    U_gt = 0.5 * tr(W_mean * (I1 + I2)) + length(mean(q_out))/2 * log(2π) - 0.5 * E_logdet_W
    Multimeta = MultiSGPMeta(method, Xu_2d,[1.0;;],kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]),kernelmatrix(kernel(θ_val),Xu_2d,[Xu_2d[1]]) * kernelmatrix(kernel(θ_val),[Xu_2d[1]],Xu_2d) + 1e-12*I,cholinv(kernelmatrix(kernel(θ_val),Xu_2d) + 1e-12*I), kernel, GPCache())
    marginals = (Marginal(q_out, false, false, nothing), Marginal(q_in, false, false, nothing), 
        Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
    U_from_node = score(AverageEnergy(), MultiSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Multimeta)
    @test typeof(U_from_node) <: Float64
    @test isapprox(U_from_node, U_gt; atol = 1e-2)

    q_out_pm = PointMass([1.5,2.0])
    q_w_pm = PointMass(W_mean)
    μ_y_pm = mean(q_out_pm)
    Ψ1_tilde = kron(C, Ψ1_approx)
    Ψ4_approx = approximate_kernel_expectation(srcubature(),(x) -> kron(C,kernelmatrix(kernel(θ_val), [x], Xu_2d)) * R_v * kron(C,kernelmatrix(kernel(θ_val), Xu_2d, [x])), q_in)
    I1 = kron(C, Ψ0_approx - tr(Kuu_inverse * Ψ2_approx))
    I2 = μ_y_pm * μ_y_pm' - μ_y_pm * μ_v' * Ψ1_tilde' - Ψ1_tilde * μ_v * μ_y_pm' + Ψ4_approx 
    U_gt = 0.5 * tr(W_mean * (I1 + I2)) + length(μ_y_pm)/2 * log(2π) - 0.5 * log(det(W_mean))
    marginals = (Marginal(q_out_pm, false, false, nothing), Marginal(q_in, false, false, nothing), 
        Marginal(q_v, false, false, nothing),Marginal(q_w_pm, false, false, nothing),Marginal(q_θ, false, false, nothing))
    U_from_node = score(AverageEnergy(), MultiSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Multimeta)
    @test typeof(U_from_node) <: Float64
    @test isapprox(U_from_node, U_gt; atol = 1e-2)
end