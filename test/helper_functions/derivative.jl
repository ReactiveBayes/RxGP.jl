using RxGP, RxInfer, ReactiveMP
using Random, Distributions, StableRNGs
using KernelFunctions, LinearAlgebra
using Test
using TestItemRunner

import RxGP: sum_diagonal_M, trace_blkmatrix

@testitem "helper_functions/derivative/UniDerivative" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    D = 1
    Nu = 10
    Xu = collect(1:Nu)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_w = GammaShapeRate(1,1)
    q_Wg = Wishart(D, diageye(D))
    μ_v, Σ_v = mean_cov(q_v)
    Rv = Σ_v + μ_v * μ_v'
    w_bar = mean(q_w)
    Wg_bar = mean(q_Wg)
    method = ghcubature(21)
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_UniSGPMeta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)
    mf = getMeanFn(Unimeta)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    KuuF = fastcholesky(kernelmatrix(kernel(θ_val),Unimeta.Xu))
    Ku_mxu = KuuF \ mxu
    mxuT_KuT = transpose(Ku_mxu)
    Ex = getEx(Unimeta)
    Dxθ = getDxθ(Unimeta)
    Cxθ_Xu = getCxθ_Xu(Unimeta)

    gt_logbackwardmess = (x,y,ω) -> begin
        if x isa Distribution
            Ψ0 = approximate_kernel_expectation(method,(x̂) -> kernelmatrix(kernel(θ_val), [x̂], [x̂]), x)[]
            Ψ1 = approximate_kernel_expectation(method,(x̂) -> kernelmatrix(kernel(θ_val), [x̂], Unimeta.Xu), x)
            Ψ2 = approximate_kernel_expectation(method,(x̂) -> kernelmatrix(kernel(θ_val), Unimeta.Xu, [x̂]) * kernelmatrix(kernel(θ_val), [x̂], Unimeta.Xu), x)
            Ψ3 = approximate_kernel_expectation(method, (x̂) -> apply_mean_fn(x̂, mf) * kernelmatrix(kernel(θ_val), [x̂], Unimeta.Xu), x)
        else
            x = x isa PointMass ? mean(x) : x
            Ψ0 = kernelmatrix(kernel(θ_val), [x], [x])[]
            Ψ1 = kernelmatrix(kernel(θ_val), [x], Unimeta.Xu)
            Ψ2 = kernelmatrix(kernel(θ_val), Unimeta.Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Unimeta.Xu) 
            Ψ3 = apply_mean_fn(x, mf) * kernelmatrix(kernel(θ_val), [x], Unimeta.Xu) 
        end

        if x isa Distribution
            Ω0 = approximate_kernel_expectation(Unimeta.method, (x̂) -> Dxθ(x̂, θ_val), x)
            Ω1 = approximate_kernel_expectation(Unimeta.method, (x̂) -> Cxθ_Xu(x̂, θ_val, Unimeta.Xu), x)
            Ω2 = approximate_kernel_expectation(Unimeta.method, (x̂) -> transpose(Cxθ_Xu(x̂, θ_val, Unimeta.Xu))*Cxθ_Xu(x̂, θ_val, Unimeta.Xu), x)
            Ω3 = approximate_kernel_expectation(Unimeta.method, (x̂) -> transpose(Cxθ_Xu(x̂, θ_val, Unimeta.Xu))*Wg_bar*Cxθ_Xu(x̂, θ_val, Unimeta.Xu), x)
            Ω4 = approximate_kernel_expectation(Unimeta.method, (x̂) -> transpose(Ex(x̂))*Wg_bar*Cxθ_Xu(x̂, θ_val, Unimeta.Xu), x)
        else
            x = x isa PointMass ? mean(x) : x
            Ω0 = Dxθ(x, θ_val)
            Ω1 = Cxθ_Xu(x, θ_val, Unimeta.Xu)
            Ω2 = transpose(Ω1) * Ω1
            Ω3 = transpose(Ω1) * Wg_bar * Ω1
            Ω4 = transpose(Ex(x)) * Wg_bar * Ω1
        end

        gt_logbackwardmess_value = (y) -> begin
            I1 = Ψ0 - tr( KuuF \ Ψ2 )
            I5 = (
                - 2 * y * jdotavx(Ψ1, ( μ_v - Ku_mxu ))
                + tr( Rv * Ψ2 ) 
                + jdotavx(mxuT_KuT, Ψ2 * Ku_mxu)
                + 2 * jdotavx(Ψ3, (μ_v - Ku_mxu))  
                - 2 * jdotavx(transpose(μ_v), Ψ2 * Ku_mxu)
            )
            -0.5 * w_bar * ( I1 + I5 ) 
        end

        gt_logbackwardmess_grad = (ω) -> begin
            G1 = Ω0 - Ω1 * (KuuF \ transpose(Ω1))
            part_A = 2 * dot(Ω4, (μ_v - Ku_mxu)) + dot((mxuT_KuT - 2*transpose(μ_v)), Ω3*Ku_mxu) + tr(Ω3 * Rv)
            part_B = 2 * dot(transpose(ω), Wg_bar * Ω1 * (μ_v - Ku_mxu))
            -0.5 * tr( Wg_bar * G1 ) - 0.5 * (part_A - part_B)
        end

        val = gt_logbackwardmess_value(y)
        grad = gt_logbackwardmess_grad(ω)
        val + grad
    end

    # data generation functions
    f(x) = sum(sin.(x.^2 .- 1) .+ cos.(x))                    # y(x) scalar for any-dimensional x
    ∇f(x) = 2 .* x .* cos.(x.^2 .- 1) .- sin.(x)              # gradient matches size(x)

    x_data_vector = [randn(D) for _ in 1:50]
    y_data_vector = [f(x) for x in x_data_vector]
    ω_data_vector = [∇f(x) for x in x_data_vector]
    gt_negllh_vector = -sum(gt_logbackwardmess(x, y, ω) for (x, y, ω) in zip(x_data_vector, y_data_vector, ω_data_vector))
    msg_negllh_vector = neg_log_backwardmess_fast(θ_val; ω_data=ω_data_vector, y_data=y_data_vector, x_data=x_data_vector, q_v=q_v, q_w=q_w, q_Wg=q_Wg, kernel=kernel, Ex=Ex, Dxθ=Dxθ, Cxθ_Xu=Cxθ_Xu, mean_fn=mean_fn, Xu=Xu)
    @test isapprox(gt_negllh_vector, msg_negllh_vector; atol=1e-6)

    x_data_pointmass = [PointMass(randn(D)) for _ in 1:50]
    y_data_pointmass = [f(mean(x)) for x in x_data_pointmass]
    ω_data_pointmass = [∇f(mean(x)) for x in x_data_pointmass]
    gt_negllh_pointmass = -sum(gt_logbackwardmess(x, y, ω) for (x, y, ω) in zip(x_data_pointmass, y_data_pointmass, ω_data_pointmass))
    msg_negllh_pointmass = neg_log_backwardmess_fast(θ_val; ω_data=ω_data_pointmass, y_data=y_data_pointmass, x_data=x_data_pointmass, q_v=q_v, q_w=q_w, q_Wg=q_Wg, kernel=kernel, Ex=Ex, Dxθ=Dxθ, Cxθ_Xu=Cxθ_Xu, mean_fn=mean_fn, Xu=Xu)
    @test isapprox(gt_negllh_pointmass, msg_negllh_pointmass; atol=1e-6)

    x_data_uncertain = [MvNormalMeanCovariance(randn(D), diageye(D)) for _ in 1:50]
    y_data_uncertain = [f(mean(x)) for x in x_data_uncertain]
    ω_data_uncertain = [∇f(mean(x)) for x in x_data_uncertain]
    gt_negllh_uncertain = -sum(gt_logbackwardmess(x, y, ω) for (x, y, ω) in zip(x_data_uncertain, y_data_uncertain, ω_data_uncertain))
    msg_negllh_uncertain = neg_log_backwardmess_fast(θ_val; ω_data=ω_data_uncertain, y_data=y_data_uncertain, x_data=x_data_uncertain, q_v=q_v, q_w=q_w, q_Wg=q_Wg, method=method, kernel=kernel, Ex=Ex, Dxθ=Dxθ, Cxθ_Xu=Cxθ_Xu, mean_fn=mean_fn, Xu=Xu)
    @test isapprox(gt_negllh_uncertain, msg_negllh_uncertain; atol=1e-6)
end