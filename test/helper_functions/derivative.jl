using RxGP, RxInfer, ReactiveMP
using Random, Distributions, StableRNGs
using KernelFunctions, LinearAlgebra
using Test
using TestItemRunner

import RxGP: sum_diagonal_M, trace_blkmatrix

@testitem "helper_functions/derivative/UniDerivative" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
    Nu = 10
    Xu = collect(1:Nu)
    xdata = collect(-5:1:5)
    ydata = sin.(xdata.^2 .- 1) .+ cos.(xdata)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_w = GammaShapeRate(1,1)
    θ_val = [1., 1.];

    μ_v, Σ_v = mean_cov(q_v)
    R_v = Σ_v + μ_v * μ_v'
    w = mean(q_w)

    Ψ0 = (x) -> kernelmatrix(kernel(θ_val), [x], [x])[1]
    Ψ1 = (x) -> kernelmatrix(kernel(θ_val), [x], Xu)
    Ψ2 = (x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu)
    Kuu_inverse = inv(kernelmatrix(kernel(θ_val),Xu))
    gt_logbackwardmess = (x,y) -> -0.5 * w * (Ψ0(x) + tr(Ψ2(x) * (R_v - Kuu_inverse)) ) + w * y * dot(Ψ1(x), μ_v)

    gt_negllh = sum(gt_logbackwardmess(x, y) for (x, y) in zip(xdata, ydata))
    Uv = cholesky(R_v).U
    approx_negllh = neg_log_backwardmess_fast(θ_val; y_data=ydata, x_data=xdata, v=μ_v, Uv=Uv, w=w, kernel=kernel, Xu=Xu)
    @test isapprox(-gt_negllh,approx_negllh;atol=1e-6)
end