module TestUnivariateNode
    using RxGP, RxInfer, ReactiveMP
    using Random, Distributions, StableRNGs
    using KernelFunctions, LinearAlgebra
    using Test
    
    @testset "UnivariateSGPNode" begin
        method = ghcubature(21)
        Nu = 10
        Xu = collect(1:Nu) #inducing points for univariate case
        kernel = (θ) -> θ[1] * with_lengthscale(SEKernel(),θ[2])
        θ_val = [1.,1.]
        q_out = Normal(1,2)
        q_w = GammaShapeRate(1,1)
        q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
        q_x = Normal(0,1)
        q_θ = PointMass(θ_val)

        μ_y = mean(q_out)
        μ_v = mean(q_v)
        R_v = μ_v * μ_v' + cov(q_v)
        E_logw = mean(log,q_w)
        Kuu_inverse = cholinv(kernelmatrix(kernel(θ_val),Xu))
        rng = StableRNG(12)
        sample_x = rand(rng,q_x,8000)
        Ψ2_func = (x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu)
        Ψ0_func = (x) -> getindex(kernelmatrix(kernel(θ_val),[x]),1)
        A_xθ = (x,θ) -> kernelmatrix(kernel(θ),[x],[x]) .- kernelmatrix(kernel(θ),[x],Xu) * inv(kernelmatrix(kernel(θ),Xu)) * kernelmatrix(kernel(θ),Xu,[x])
        B_xθ = (x,θ) -> kernelmatrix(kernel(θ), [x], Xu)
        A_x = (x) -> A_xθ(x,θ_val)
        B_x = (x) -> B_xθ(x,θ_val)

        Ψ0_gt = mean(Ψ0_func.(sample_x))
        Ψ1_gt = mean(B_x.(sample_x))
        Ψ2_gt = mean(Ψ2_func.(sample_x)) + 1e-8*I

        Kuu = kernelmatrix(kernel(θ_val), Xu)
        KuuL = fastcholesky(Kuu).L
        Uv = cholesky(R_v).U
        Ψ0 = [1.0;;]
        Ψ1_trans = kernelmatrix(kernel(θ_val),Xu,[1.])
        Ψ2 = kernelmatrix(kernel(θ_val),Xu,[1.]) * kernelmatrix(kernel(θ_val),[1.],Xu);

        Ψ0_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), [x], [x]),q_x)[]
        Ψ1_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), [x], Xu),q_x)
        Ψ2_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu), q_x) + 1e-8*I

        Unimeta = UniSGPMeta(method,Xu,Ψ0,Ψ1_trans,Ψ2,KuuL,kernel,Uv,0,1)
        @testset "Test GPMeta" begin
            #check if we can get the elements of meta
            @test getInducingInput(Unimeta) == Xu
            @test getKernel(Unimeta) == kernel
            @test typeof(getKernel(Unimeta)) <: Function
            @test getmethod(Unimeta) == method
            @test getΨ0(Unimeta) == Ψ0 
            @test getΨ1_trans(Unimeta) == Ψ1_trans
            @test getΨ2(Unimeta) == Ψ2
            @test getUv(Unimeta) == Uv # Cholesky upper triangular of Rv = μ_v * μ_v' + Σ_v
            @test getKuuL(Unimeta) == KuuL
            @test getcounter(Unimeta) == 0
            @test getN(Unimeta) == 1
        end

        @testset "Test out rule" begin
            gt_mean_y =  getindex(Ψ1_approx * mean(q_v),1)
            gt_var_y = inv(mean(q_w))

            #q_in is a distribution
            ν_y_1 =  @call_rule UniSGP(:out, Marginalisation) (q_in = q_x, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
            @test typeof(ν_y_1) <: UnivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_y_1), gt_mean_y ; atol=1e-7)
            @test isapprox(var(ν_y_1), gt_var_y)

            #q_in is a PointMass
            Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            ν_y_2 = @call_rule UniSGP(:out, Marginalisation) (q_in = PointMass(1.0), q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
            @test typeof(ν_y_2) <: UnivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_y_2), getindex(Ψ1 * mean(q_v),1))
            @test isapprox(var(ν_y_2), gt_var_y)
        end

        @testset "Test in rule" begin
            gt_logbackwardmess_x = (x) -> getindex(-0.5 * mean(q_w) * (A_x(x) + B_x(x) * R_v * B_x(x)' - 2* μ_y * B_x(x)*μ_v),1)
            ν_x = @call_rule UniSGP(:in, Marginalisation) (q_out = q_out, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
            @test typeof(ν_x) <: ContinuousUnivariateLogPdf
            @test isapprox(logpdf(ν_x,1.0), gt_logbackwardmess_x(1.0))
            @test isapprox(logpdf(ν_x,sqrt(2)), gt_logbackwardmess_x(sqrt(2)))
            @test isapprox(logpdf(ν_x,4.2), gt_logbackwardmess_x(4.2))
        end

        @testset "Test v rule" begin
            #q_out, q_in are Normal
            ν_v_1 = @call_rule UniSGP(:v, Marginalisation) (q_out = q_out, q_in = q_x, q_w = q_w, q_θ = q_θ, meta = Unimeta)
            gt_mean_v_1 = vcat(inv(Ψ2_approx) * Ψ1_approx' * μ_y...) 
            gt_cov_v_1 = inv(mean(q_w) * Ψ2_approx)
            @test typeof(ν_v_1) <: BufferUniSGP
            @test typeof(ν_v_1.qv) <: MultivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_v_1.qv), gt_mean_v_1)
            @test isapprox(cov(ν_v_1.qv), gt_cov_v_1)

            #q_out, q_in are PointMass
            Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) 
            ν_v_2 = @call_rule UniSGP(:v, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.0), q_w = q_w, q_θ = q_θ, meta = Unimeta)
            gt_mean_v_2 = vcat(cholinv(Ψ2) * Ψ1' * 2...) 
            gt_cov_v_2 = cholinv(mean(q_w) * Ψ2)
            @test typeof(ν_v_2) <: BufferUniSGP
            @test typeof(ν_v_2.qv) <: MultivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_v_2.qv), gt_mean_v_2)
            @test isapprox(cov(ν_v_2.qv), gt_cov_v_2)

            #q_out is Normal, q_in is PointMass
            ν_v_3 = @call_rule UniSGP(:v, Marginalisation) (q_out = q_out, q_in = PointMass(1.0), q_w = q_w, q_θ = q_θ, meta = Unimeta)
            gt_mean_v_3 = vcat(cholinv(Ψ2) * Ψ1' * μ_y...) 
            gt_cov_v_3 = cholinv(mean(q_w) * Ψ2)
            @test typeof(ν_v_3) <: BufferUniSGP
            @test typeof(ν_v_3.qv) <: MultivariateGaussianDistributionsFamily
            @test isapprox(mean(ν_v_3.qv), gt_mean_v_3)
            @test isapprox(cov(ν_v_3.qv), gt_cov_v_3);
        end

        @testset "Test w rule" begin
            #q_out, q_in :: Normal
            I1 = Ψ0_approx - tr(Kuu_inverse * Ψ2_approx)
            I2 = mean(q_out)^2 + var(q_out) - 2*mean(q_out)*getindex(Ψ1_approx*mean(q_v),1) + tr(R_v*Ψ2_approx)
            rate_gt = 0.5 * (I1 + I2)
            ν_w_1 = @call_rule UniSGP(:w, Marginalisation) (q_out = q_out, q_in = q_x, q_v = q_v, q_θ = q_θ, meta = Unimeta)
            @test typeof(ν_w_1) <: GammaDistributionsFamily
            @test shape(ν_w_1) == 1.5
            @test isapprox(rate(ν_w_1), rate_gt; atol=1e-5)

            #q_out, q_in :: PointMass
            Ψ0 =  getindex(kernelmatrix(kernel(θ_val), [1.0], [1.0]),1)
            Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-8*I 
            I1 = Ψ0 - tr(Kuu_inverse * Ψ2)
            I2 = 2.0^2 - 2*2.0*getindex(Ψ1 * mean(q_v),1) + tr(R_v * Ψ2)
            ν_w_2 = @call_rule UniSGP(:w, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.0), q_v = q_v, q_θ = q_θ, meta = Unimeta)
            @test typeof(ν_w_2) <: GammaDistributionsFamily
            @test shape(ν_w_2) == 1.5
            @test isapprox(rate(ν_w_2),0.5 * (I1 + I2); atol=1e-5)

            #q_out::Normal, q_in::PointMass
            I2 = mean(q_out)^2 + var(q_out) - 2*mean(q_out)*getindex(Ψ1*mean(q_v),1) + tr(R_v*Ψ2)
            ν_w_3 = @call_rule UniSGP(:w, Marginalisation) (q_out = q_out, q_in = PointMass(1.0), q_v = q_v, q_θ = q_θ, meta = Unimeta)
            @test typeof(ν_w_3) <: GammaDistributionsFamily
            @test shape(ν_w_3) == 1.5
            @test isapprox(rate(ν_w_3),0.5 * (I1 + I2); atol=1e-5)
        end

        @testset "Test θ rule" begin
            Kuu_inverse_θ = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu))

            #q_out, q_in :: Normal 
            Ψ0_θ = (θ) -> approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], [x]),q_x)[]
            Ψ1_θ = (θ) -> approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], Xu),q_x)
            Ψ2_θ = (θ) -> approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_x)
            gt_logbackwardmess_θ = (θ) -> -0.5 * mean(q_w) * (Ψ0_θ(θ) + tr(Ψ2_θ(θ) * (R_v - Kuu_inverse_θ(θ))) ) + mean(q_w) * mean(q_out) * getindex(Ψ1_θ(θ) * μ_v,1)
            ν_θ_1 = @call_rule UniSGP(:θ, Marginalisation) (q_out = q_out, q_in = q_x, q_v = q_v, q_w = q_w, meta = Unimeta)
            @test typeof(ν_θ_1) <: ContinuousMultivariateLogPdf
            @test isapprox(logpdf(ν_θ_1,[1,2]), gt_logbackwardmess_θ([1,2]);atol=1e-7)
            @test isapprox(logpdf(ν_θ_1,[0.5,1.4]), gt_logbackwardmess_θ([0.5,1.4]); atol=1e-7)

            #q_out::Normal, q_in::PointMass
            Ψ0_θ_2 = (θ) -> kernelmatrix(kernel(θ), [1.0], [1.0])[1]
            Ψ1_θ_2 = (θ) -> kernelmatrix(kernel(θ), [1.0], Xu)
            Ψ2_θ_2 = (θ) -> kernelmatrix(kernel(θ), Xu, [1.0]) * kernelmatrix(kernel(θ), [1.0], Xu) 
            gt_logbackwardmess_θ_2 = (θ) -> -0.5 * mean(q_w) * (Ψ0_θ_2(θ) + tr(Ψ2_θ_2(θ) * (R_v - Kuu_inverse_θ(θ))) ) + mean(q_w) * mean(q_out) * getindex(Ψ1_θ_2(θ) * μ_v,1)
            ν_θ_2 = @call_rule UniSGP(:θ, Marginalisation) (q_out = q_out, q_in = PointMass(1.0), q_v = q_v, q_w = q_w, meta = Unimeta)
            @test typeof(ν_θ_2) <: ContinuousMultivariateLogPdf
            @test isapprox(logpdf(ν_θ_2,[1,2]), gt_logbackwardmess_θ_2([1,2]);atol=1e-9)
            @test isapprox(logpdf(ν_θ_2,[0.5,1.4]), gt_logbackwardmess_θ_2([0.5,1.4]);atol = 1e-9)

            #q_out, q_in::PointMass
            gt_logbackwardmess_θ_3 = (θ) -> -0.5 * mean(q_w) * (Ψ0_θ_2(θ) + tr(Ψ2_θ_2(θ) * (R_v - Kuu_inverse_θ(θ))) ) + mean(q_w) * 2.0 * getindex(Ψ1_θ_2(θ) * μ_v,1)
            ν_θ_3 = @call_rule UniSGP(:θ, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.0), q_v = q_v, q_w = q_w, meta = Unimeta)
            @test typeof(ν_θ_3) <: ContinuousMultivariateLogPdf
            @test isapprox(logpdf(ν_θ_3,[1,2]), gt_logbackwardmess_θ_3([1,2]);atol=1e-9)
            @test isapprox(logpdf(ν_θ_3,[0.5,1.4]), gt_logbackwardmess_θ_3([0.5,1.4]);atol=1e-9)
        end

        @testset "Test average energy" begin
            #q_out, q_in :: PointMass, q_w :: Gamma
            Ψ0_1 =  getindex(kernelmatrix(kernel(θ_val), [1.0], [1.0]),1)
            Ψ1_1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2_1 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-7*I 
            I1_1 = Ψ0_1 - tr(Kuu_inverse * Ψ2_1)
            I2_1 = 2.0^2 - 2*2.0*getindex(Ψ1_1 * mean(q_v),1) + tr(R_v * Ψ2_1)
            U_gt = 0.5 * log(2π) - 0.5 * E_logw + 0.5 * mean(q_w) * (I1_1 + I2_1)
            
            marginals = (Marginal(PointMass(2.0), false, false, nothing), Marginal(PointMass(1.0), false, false, nothing), 
                        Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt; atol = 1e-5)

            #q_out :: Normal, q_in :: PointMass, q_w :: Gamma
            Ψ0_2 =  getindex(kernelmatrix(kernel(θ_val), [1.0], [1.0]),1)
            Ψ1_2 = kernelmatrix(kernel(θ_val), [1.0], Xu)
            Ψ2_2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-7*I 
            I1_2 = Ψ0_2 - tr(Kuu_inverse * Ψ2_2)
            I2_2 = mean(q_out)^2 + var(q_out)- 2*mean(q_out)*getindex(Ψ1_2 * mean(q_v),1) + tr(R_v * Ψ2_2)
            U_gt = 0.5 * log(2π) - 0.5 * E_logw + 0.5 * mean(q_w) * (I1_2 + I2_2)
            
            marginals = (Marginal(q_out, false, false, nothing), Marginal(PointMass(1.0), false, false, nothing), 
                        Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt; atol = 1e-5)

            #q_out, q_in :: Normal, q_w :: Gamma
            I1_3 = Ψ0_approx - tr(Kuu_inverse * Ψ2_approx)
            I2_3 = mean(q_out)^2 + var(q_out)- 2*mean(q_out)*getindex(Ψ1_approx * mean(q_v),1) + tr(R_v * Ψ2_approx)
            U_gt = 0.5 * log(2π) - 0.5 * E_logw + 0.5 * mean(q_w) * (I1_3 + I2_3)
            
            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_x, false, false, nothing), 
                        Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt; atol=1e-5)

            #q_out, q_in :: Normal, q_w :: PointMass
            w = 5.0
            I1_4 = Ψ0_approx - tr(Kuu_inverse * Ψ2_approx)
            I2_4 = mean(q_out)^2 + var(q_out)- 2*mean(q_out)*getindex(Ψ1_approx * mean(q_v),1) + tr(R_v * Ψ2_approx)
            U_gt = 0.5 * log(2π) - 0.5 * log(w) + 0.5 * w * (I1_4 + I2_4)
            
            marginals = (Marginal(q_out, false, false, nothing), Marginal(q_x, false, false, nothing), 
                        Marginal(q_v, false, false, nothing),Marginal(PointMass(w), false, false, nothing),Marginal(q_θ, false, false, nothing))
            U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta)
            @test typeof(U_from_node) <: Float64
            @test isapprox(U_from_node, U_gt;atol=1e-5);
        end

    end 




end #end module