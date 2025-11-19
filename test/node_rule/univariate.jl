@testitem "node_rule/univariate/Test GPMeta" begin
    using Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, StatsFuns, RxInfer, ReactiveMP
    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D=1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> zero(x)
    Unimeta = get_GP_meta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)
    Kuu = kernelmatrix(kernel(θ_val), Xu) + 1e-8 * I
    KuuF = fastcholesky(Kuu)
    x_dummy = zeros(D)
    Ψ0 = kernelmatrix(kernel(θ_val), [x_dummy])
    Ψ1_trans = kernelmatrix(kernel(θ_val),Xu,[x_dummy])
    Ψ2 = kernelmatrix(kernel(θ_val),Xu,[x_dummy]) * kernelmatrix(kernel(θ_val),[x_dummy],Xu)
    @test getInducingInput(Unimeta) == Xu
    @test getKernel(Unimeta) == kernel
    @test typeof(getKernel(Unimeta)) <: Function
    @test getmethod(Unimeta) == method
    @test getΨ0(Unimeta) == Ψ0 
    @test getΨ1_trans(Unimeta) == Ψ1_trans
    @test getΨ2(Unimeta) == Ψ2
    @test getKuuF(Unimeta) == KuuF
    @test getcounter(Unimeta) == 0
    @test getN(Unimeta) == 1
end

@testitem "node_rule/univariate/Test out rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test

    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D=1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_GP_meta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)
    x_dummy = zeros(D)

    q_out = Normal(1,2)
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_in = Normal(0,1)
    q_θ = PointMass(θ_val)
    μ_y = mean(q_out)
    μ_in = mean(q_in)
    μ_v = mean(q_v)
    R_v = μ_v * μ_v' + cov(q_v)
    
    mf = getMeanFn(Unimeta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu

    Kuu_inverse = cholinv(kernelmatrix(kernel(θ_val),Xu))
    Ψ1 = kernelmatrix(kernel(θ_val), [μ_in], Xu)
    Ψ1_approx = approximate_kernel_expectation(method, (x) -> kernelmatrix(kernel(θ_val), [x], Xu), q_in)
    approximate_kernel_expectation!(Unimeta.Ψ1_trans, method, (x) -> kernelmatrix(kernel(θ_val), Xu, [x]), q_in)

    gt_mean_y =  getindex(mx + jdotavx(Ψ1, μ_v) - jdotavx(Ψ1, Ku_mxu), 1)
    gt_mean_y_approx =  getindex(mx + jdotavx(Ψ1_approx, μ_v) - jdotavx(Ψ1_approx, Ku_mxu), 1)
    gt_var_y = inv(mean(q_w))
    ν_y_1 =  @call_rule UniSGP(:out, Marginalisation) (q_in = q_in, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_y_1) <: UnivariateGaussianDistributionsFamily
    @test isapprox(mean(ν_y_1), gt_mean_y_approx ; atol=1e-7)
    @test isapprox(var(ν_y_1), gt_var_y)

    ν_y_2 = @call_rule UniSGP(:out, Marginalisation) (q_in = PointMass(0.0), q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_y_2) <: UnivariateGaussianDistributionsFamily
    @test isapprox(mean(ν_y_2), gt_mean_y ; atol=1e-7)
    @test isapprox(var(ν_y_2), gt_var_y)
end

@testitem "node_rule/univariate/Test in rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D=1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_GP_meta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)
    x_dummy = zeros(D)

    q_out = Normal(1,2)
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_in = Normal(0,1)
    q_θ = PointMass(θ_val)
    w_bar = mean(q_w)
    μ_in = mean(q_in)
    μ_y = mean(q_out)
    μ_v = mean(q_v)
    R_v = μ_v * μ_v' + cov(q_v)
    
    mf = getMeanFn(Unimeta)
    mx = (x) -> apply_mean_fn(x, mf)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu

    kxx = (x) -> kernelmatrix(kernel(θ_val),[x])
    kxu = (x) -> kernelmatrix(kernel(θ_val),[x], Xu)
    A = (x) -> kxx(x) - kxu(x) * (Unimeta.KuuF \ transpose(kxu(x)))
    B = (x) -> kxu(x)

    partA = (x) -> -0.5 * w_bar * A(x)[1]
    partB = (x) -> w_bar * μ_y * ( 
        mx(x) 
        + dot(B(x), μ_v) 
        - dot(B(x), Ku_mxu)
        )
    partC = (x) -> -0.5 * w_bar * ( 
        mx(x)^2 
        + dot(B(x), R_v * B(x)')
        + dot(Ku_mxu' * B(x)', B(x) * Ku_mxu)
        + 2 * dot(mx(x), B(x) * μ_v)
        - 2 * dot(mx(x), B(x) * Ku_mxu)
        - 2 * dot(μ_v' * B(x)', B(x) * Ku_mxu)
        )

    gt_logbackwardmess_x = (x) -> partA(x) + partB(x) + partC(x)

    ν_x = @call_rule UniSGP(:in, Marginalisation) (q_out = q_out, q_v = q_v, q_w = q_w, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_x) <: ContinuousUnivariateLogPdf
    @test isapprox(logpdf(ν_x,1.0), gt_logbackwardmess_x(1.0))
    @test isapprox(logpdf(ν_x,sqrt(2)), gt_logbackwardmess_x(sqrt(2)))
    @test isapprox(logpdf(ν_x,4.2), gt_logbackwardmess_x(4.2))
end

@testitem "node_rule/univariate/Test v rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D=1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_GP_meta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)
    x_dummy = zeros(D)

    q_out = Normal(1,2)
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_in = Normal(0,1)
    q_θ = PointMass(θ_val)
    w_bar = mean(q_w)
    μ_in = mean(q_in)
    μ_y = mean(q_out)
    μ_v = mean(q_v)
    R_v = μ_v * μ_v' + cov(q_v)
    
    mf = getMeanFn(Unimeta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu
    Ψ1_trans_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), Xu, [x]), q_in)
    Ψ2_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu), q_in) + 1e-8*I
    
    ν_v_1 = @call_rule UniSGP(:v, Marginalisation) (q_out = q_out, q_in = q_in, q_w = q_w, q_θ = q_θ, meta = Unimeta)
    gt_mean_v_1 = vec((fastcholesky(Ψ2_approx) \ Ψ1_trans_approx) * (μ_y - mx) + Ku_mxu)
    gt_cov_v_1 = cholinv(w_bar * Ψ2_approx)
    @test typeof(ν_v_1) <: BufferUniSGP
    @test typeof(ν_v_1.qv) <: MultivariateGaussianDistributionsFamily
    @test isapprox(mean(ν_v_1.qv), gt_mean_v_1)
    @test isapprox(cov(ν_v_1.qv), gt_cov_v_1)

    Ψ1_trans = kernelmatrix(kernel(θ_val), Xu, [1.0])
    Ψ2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-8*I
    mx = apply_mean_fn([1.0], mf)
    ν_v_2 = @call_rule UniSGP(:v, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.0), q_w = q_w, q_θ = q_θ, meta = Unimeta)
    gt_mean_v_2 = vec((fastcholesky(Ψ2) \ Ψ1_trans) * (2.0 - mx) + Ku_mxu)
    gt_cov_v_2 = cholinv(w_bar * Ψ2)
    @test typeof(ν_v_2) <: BufferUniSGP
    @test typeof(ν_v_2.qv) <: MultivariateGaussianDistributionsFamily
    @test isapprox(mean(ν_v_2.qv), gt_mean_v_2)
    @test isapprox(cov(ν_v_2.qv), gt_cov_v_2)

    ν_v_3 = @call_rule UniSGP(:v, Marginalisation) (q_out = q_out, q_in = PointMass(1.0), q_w = q_w, q_θ = q_θ, meta = Unimeta)
    gt_mean_v_3 = vec((fastcholesky(Ψ2) \ Ψ1_trans) * (μ_y - mx) + Ku_mxu)
    gt_cov_v_3 = cholinv(w_bar * Ψ2)
    @test typeof(ν_v_3) <: BufferUniSGP
    @test typeof(ν_v_3.qv) <: MultivariateGaussianDistributionsFamily
    @test isapprox(mean(ν_v_3.qv), gt_mean_v_3)
    @test isapprox(cov(ν_v_3.qv), gt_cov_v_3)
end

@testitem "node_rule/univariate/Test w rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D=1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_GP_meta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)
    x_dummy = zeros(D)

    q_out = Normal(1,2)
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_in = Normal(0,1)
    q_θ = PointMass(θ_val)
    w_bar = mean(q_w)
    μ_in = mean(q_in)
    μ_y = mean(q_out)
    μ_v = mean(q_v)
    R_v = μ_v * μ_v' + cov(q_v)
    
    mf = getMeanFn(Unimeta)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu

    Ψ0_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), [x], [x]), q_in)[1]
    Ψ1_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), [x], Xu), q_in)
    Ψ1_trans_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), Xu, [x]), q_in)
    Ψ2_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu), q_in) + 1e-8*I
    Ψ0 =  getindex(kernelmatrix(kernel(θ_val), [1.0], [1.0]),1)
    Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
    Ψ1_trans = kernelmatrix(kernel(θ_val), Xu, [1.0])
    Ψ2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-8*I 

    mx = apply_mean_fn(μ_in, mf)
    α = Unimeta.KuuF.L \ Ψ1_trans_approx
    I1 = Ψ0_approx - jdotavx(α,α) # I1 = Ψ0_approx - tr(Unimeta.KuuF \ Ψ2_approx)
    I4 = (
        μ_y^2
        + var(q_out)
        - 2 * μ_y * ( mx + getindex(Ψ1_approx * ( μ_v - Ku_mxu ), 1) )
        + mx^2 
        + tr(R_v * Ψ2_approx)
        + transpose(Ku_mxu) * Ψ2_approx * Ku_mxu
        + 2 * mx * getindex(Ψ1_approx * μ_v, 1) 
        - 2 * mx * getindex(Ψ1_approx * Ku_mxu, 1) 
        - 2 * transpose(μ_v) * Ψ2_approx * Ku_mxu
    )
    rate_gt = 0.5*(I1 + I4)
    ν_w_1 = @call_rule UniSGP(:w, Marginalisation) (q_out = q_out, q_in = q_in, q_v = q_v, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_w_1) <: GammaDistributionsFamily
    @test shape(ν_w_1) == 1.5
    @test isapprox(rate(ν_w_1), rate_gt; atol=1e-5)

    mx = apply_mean_fn(1.0, mf)
    α = Unimeta.KuuF.L \ Ψ1_trans
    I1 = Ψ0 - jdotavx(α,α) # I1 = Ψ0 - tr(Unimeta.KuuF \ Ψ2)
    I4 = (
        2^2
        + 0
        - 2 * 2 * ( mx + getindex(Ψ1 * ( μ_v - Ku_mxu ), 1) )
        + mx^2 
        + tr(R_v * Ψ2)
        + transpose(Ku_mxu) * Ψ2 * Ku_mxu
        + 2 * mx * getindex(Ψ1 * μ_v, 1) 
        - 2 * mx * getindex(Ψ1 * Ku_mxu, 1) 
        - 2 * transpose(μ_v) * Ψ2 * Ku_mxu
    )
    ν_w_2 = @call_rule UniSGP(:w, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.0), q_v = q_v, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_w_2) <: GammaDistributionsFamily
    @test shape(ν_w_2) == 1.5
    @test isapprox(rate(ν_w_2),0.5 * (I1 + I4); atol=1e-5)

    I4 = (
        μ_y^2
        + var(q_out)
        - 2 * μ_y * ( mx + getindex(Ψ1 * ( μ_v - Ku_mxu ), 1) )
        + mx^2 
        + tr(R_v * Ψ2)
        + transpose(Ku_mxu) * Ψ2 * Ku_mxu
        + 2 * mx * getindex(Ψ1 * μ_v, 1) 
        - 2 * mx * getindex(Ψ1 * Ku_mxu, 1) 
        - 2 * transpose(μ_v) * Ψ2 * Ku_mxu
    )
    ν_w_3 = @call_rule UniSGP(:w, Marginalisation) (q_out = q_out, q_in = PointMass(1.0), q_v = q_v, q_θ = q_θ, meta = Unimeta)
    @test typeof(ν_w_3) <: GammaDistributionsFamily
    @test shape(ν_w_3) == 1.5
    @test isapprox(rate(ν_w_3),0.5 * (I1 + I4); atol=1e-5)
end

@testitem "node_rule/univariate/Test θ rule" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D=1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_GP_meta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)
    x_dummy = zeros(D)

    q_out = Normal(1,2)
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_in = Normal(0,1)
    q_θ = PointMass(θ_val)
    w_bar = mean(q_w)
    μ_in = mean(q_in)
    μ_y = mean(q_out)
    μ_v = mean(q_v)
    R_v = μ_v * μ_v' + cov(q_v)
    
    mf = getMeanFn(Unimeta)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)

    Ψ0_θ_approx = (θ) -> approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[]
    Ψ1_θ_approx = (θ) -> approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
    Ψ2_θ_approx = (θ) -> approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in)
    Ψ0_θ = (θ) -> kernelmatrix(kernel(θ), [1.0], [1.0])[1]
    Ψ1_θ = (θ) -> kernelmatrix(kernel(θ), [1.0], Xu)
    Ψ2_θ = (θ) -> kernelmatrix(kernel(θ), Xu, [1.0]) * kernelmatrix(kernel(θ), [1.0], Xu) 
    KuuF = (θ) -> fastcholesky(kernelmatrix(kernel(θ),Xu))
    Ku_mxu = (θ) -> KuuF(θ) \ mxu
    mxuT_KuT = (θ) -> transpose(Ku_mxu(θ))

    mx = apply_mean_fn(μ_in, mf)
    I1_θ = (θ) -> Ψ0_θ_approx(θ) - tr( KuuF(θ) \ Ψ2_θ_approx(θ) )
    I5_θ = (θ) -> (
        - 2 * μ_y * dot(Ψ1_θ_approx(θ), ( μ_v - Ku_mxu(θ) ))
        + tr( R_v * Ψ2_θ_approx(θ) )
        + dot(mxuT_KuT(θ), Ψ2_θ_approx(θ) * Ku_mxu(θ))
        + 2 * mx * dot(Ψ1_θ_approx(θ), μ_v)  
        - 2 * mx * dot(Ψ1_θ_approx(θ), Ku_mxu(θ)) 
        - 2 * dot(transpose(μ_v), Ψ2_θ_approx(θ) * Ku_mxu(θ))
    )
    gt_logbackwardmess_θ = (θ) -> -0.5 * w_bar * (I1_θ(θ) + I5_θ(θ))
    ν_θ_1 = @call_rule UniSGP(:θ, Marginalisation) (q_out = q_out, q_in = q_in, q_v = q_v, q_w = q_w, meta = Unimeta)
    @test typeof(ν_θ_1) <: ContinuousMultivariateLogPdf
    @test isapprox(logpdf(ν_θ_1,[1,2]), gt_logbackwardmess_θ([1,2]);atol=1e-7)
    @test isapprox(logpdf(ν_θ_1,[0.5,1.4]), gt_logbackwardmess_θ([0.5,1.4]); atol=1e-7)

    mx = apply_mean_fn(1, mf)
    I1_θ = (θ) -> Ψ0_θ(θ) - tr( KuuF(θ) \ Ψ2_θ(θ) )
    I5_θ = (θ) -> (
        - 2 * μ_y * dot(Ψ1_θ(θ), ( μ_v - Ku_mxu(θ) ))
        + tr( R_v * Ψ2_θ(θ) )
        + dot(mxuT_KuT(θ), Ψ2_θ(θ) * Ku_mxu(θ))
        + 2 * mx * dot(Ψ1_θ(θ), μ_v)  
        - 2 * mx * dot(Ψ1_θ(θ), Ku_mxu(θ)) 
        - 2 * dot(transpose(μ_v), Ψ2_θ(θ) * Ku_mxu(θ))
    )
    gt_logbackwardmess_θ = (θ) -> -0.5 * w_bar * (I1_θ(θ) + I5_θ(θ))
    ν_θ = @call_rule UniSGP(:θ, Marginalisation) (q_out = q_out, q_in = PointMass(1.0), q_v = q_v, q_w = q_w, meta = Unimeta)
    @test typeof(ν_θ) <: ContinuousMultivariateLogPdf
    @test isapprox(logpdf(ν_θ,[1,2]), gt_logbackwardmess_θ([1,2]);atol=1e-9)
    @test isapprox(logpdf(ν_θ,[0.5,1.4]), gt_logbackwardmess_θ([0.5,1.4]);atol = 1e-9)

    mx = apply_mean_fn(1, mf)
    I1_θ = (θ) -> Ψ0_θ(θ) - tr( KuuF(θ) \ Ψ2_θ(θ) )
    I5_θ = (θ) -> (
        - 2 * 2 * dot(Ψ1_θ(θ), ( μ_v - Ku_mxu(θ) ))
        + tr( R_v * Ψ2_θ(θ) )
        + dot(mxuT_KuT(θ), Ψ2_θ(θ) * Ku_mxu(θ))
        + 2 * mx * dot(Ψ1_θ(θ), μ_v)  
        - 2 * mx * dot(Ψ1_θ(θ), Ku_mxu(θ)) 
        - 2 * dot(transpose(μ_v), Ψ2_θ(θ) * Ku_mxu(θ))
    )
    gt_logbackwardmess_θ_3 = (θ) -> -0.5 * w_bar * (I1_θ(θ) + I5_θ(θ))
    ν_θ_3 = @call_rule UniSGP(:θ, Marginalisation) (q_out = PointMass(2.0), q_in = PointMass(1.0), q_v = q_v, q_w = q_w, meta = Unimeta)
    @test typeof(ν_θ_3) <: ContinuousMultivariateLogPdf
    @test isapprox(logpdf(ν_θ_3,[1,2]), gt_logbackwardmess_θ_3([1,2]);atol=1e-9)
    @test isapprox(logpdf(ν_θ_3,[0.5,1.4]), gt_logbackwardmess_θ_3([0.5,1.4]);atol=1e-9)
end

@testitem "node_rule/univariate/Test average energy" begin
    using RxGP, RxInfer, ReactiveMP, Random, Distributions, StableRNGs, KernelFunctions, LinearAlgebra, Test
    method = ghcubature(21)
    Nu = 10
    Xu = collect(1:Nu)
    D=1
    kernel, θ_val, dim_θ = get_simple_kernel_and_params(D; kernel_spec=:SE)
    mean_fn = x -> sum(x)
    Unimeta = get_GP_meta(D; method=method, mean_fn=mean_fn, kernel=kernel, kernel_spec=:SE, mode=:AN, independent_SE_lengthscales=false, Xu=Xu, θ=θ_val)
    x_dummy = zeros(D)

    q_out = Normal(1,2)
    q_w = GammaShapeRate(1,1)
    q_v = MvNormalMeanCovariance(rand(Nu) |> (x) -> sin.(x), diageye(Nu))
    q_in = Normal(0,1)
    q_θ = PointMass(θ_val)
    w_bar = mean(q_w)
    E_logw = mean(log,q_w)
    μ_in = mean(q_in)
    μ_y = mean(q_out)
    μ_v = mean(q_v)
    R_v = μ_v * μ_v' + cov(q_v)
    
    mf = getMeanFn(Unimeta)
    mx = apply_mean_fn(μ_in, mf)
    mxu = apply_mean_fn.(Unimeta.Xu, mf)
    Ku_mxu = Unimeta.KuuF \ mxu

    Ψ0_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), [x], [x]),q_in)[]
    Ψ1_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), [x], Xu),q_in)
    Ψ1_trans_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), Xu, [x]),q_in)
    Ψ2_approx = approximate_kernel_expectation(method,(x) -> kernelmatrix(kernel(θ_val), Xu, [x]) * kernelmatrix(kernel(θ_val), [x], Xu), q_in) + 1e-8*I
    Ψ0 =  getindex(kernelmatrix(kernel(θ_val), [1.0], [1.0]),1)
    Ψ1 = kernelmatrix(kernel(θ_val), [1.0], Xu)
    Ψ1_trans = kernelmatrix(kernel(θ_val), Xu, [1.0])
    Ψ2 = kernelmatrix(kernel(θ_val), Xu, [1.0]) * kernelmatrix(kernel(θ_val), [1.0], Xu) + 1e-7*I 

    mx = apply_mean_fn(1.0, mf)
    α = Unimeta.KuuF.L \ Ψ1_trans
    I1 = Ψ0 - jdotavx(α,α) # I1 = Ψ0 - tr(Unimeta.KuuF \ Ψ2)
    I4 = (
        2^2
        + 0
        - 2 * 2 * ( mx + getindex(Ψ1 * ( μ_v - Ku_mxu ), 1) )
        + mx^2 
        + tr(R_v * Ψ2)
        + transpose(Ku_mxu) * Ψ2 * Ku_mxu
        + 2 * mx * getindex(Ψ1 * μ_v, 1) 
        - 2 * mx * getindex(Ψ1 * Ku_mxu, 1) 
        - 2 * transpose(μ_v) * Ψ2 * Ku_mxu
    )
    U_gt = 0.5 * log(2π) - 0.5 * E_logw + 0.5 * mean(q_w) * (I1 + I4)
    marginals = (Marginal(PointMass(2.0), false, false, nothing), Marginal(PointMass(1.0), false, false, nothing), 
                Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
    U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta)
    @test typeof(U_from_node) <: Float64
    @test isapprox(U_from_node, U_gt; atol = 1e-5)

    mx = apply_mean_fn(1.0, mf)
    α = Unimeta.KuuF.L \ Ψ1_trans
    I1 = Ψ0 - jdotavx(α,α) # I1 = Ψ0 - tr(Unimeta.KuuF \ Ψ2)
    I4 = (
        μ_y^2
        + var(q_out)
        - 2 * μ_y * ( mx + getindex(Ψ1 * ( μ_v - Ku_mxu ), 1) )
        + mx^2 
        + tr(R_v * Ψ2)
        + transpose(Ku_mxu) * Ψ2 * Ku_mxu
        + 2 * mx * getindex(Ψ1 * μ_v, 1) 
        - 2 * mx * getindex(Ψ1 * Ku_mxu, 1) 
        - 2 * transpose(μ_v) * Ψ2 * Ku_mxu
    )
    U_gt = 0.5 * log(2π) - 0.5 * E_logw + 0.5 * mean(q_w) * (I1 + I4)
    marginals = (Marginal(q_out, false, false, nothing), Marginal(PointMass(1.0), false, false, nothing), 
                Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
    U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta)
    @test typeof(U_from_node) <: Float64
    @test isapprox(U_from_node, U_gt; atol = 1e-5)

    
    
    
    
    mx = apply_mean_fn(μ_in, mf)
    α = Unimeta.KuuF.L \ Ψ1_trans_approx
    I1 = Ψ0_approx - jdotavx(α,α) # I1 = Ψ0_approx - tr(Unimeta.KuuF \ Ψ2_approx)
    I4 = (
        μ_y^2
        + var(q_out)
        - 2 * μ_y * ( mx + getindex(Ψ1_approx * ( μ_v - Ku_mxu ), 1) )
        + mx^2 
        + tr(R_v * Ψ2_approx)
        + transpose(Ku_mxu) * Ψ2_approx * Ku_mxu
        + 2 * mx * getindex(Ψ1_approx * μ_v, 1) 
        - 2 * mx * getindex(Ψ1_approx * Ku_mxu, 1) 
        - 2 * transpose(μ_v) * Ψ2_approx * Ku_mxu
    )
    
    I1_3 = Ψ0_approx - tr(Unimeta.KuuF \ Ψ2_approx)
    I2_3 = mean(q_out)^2 + var(q_out)- 2*mean(q_out)*getindex(Ψ1_approx * mean(q_v),1) + tr(R_v * Ψ2_approx)
    U_gt = 0.5 * log(2π) - 0.5 * E_logw + 0.5 * mean(q_w) * (I1 + I4)
    marginals = (Marginal(q_out, false, false, nothing), Marginal(q_in, false, false, nothing), 
                Marginal(q_v, false, false, nothing),Marginal(q_w, false, false, nothing),Marginal(q_θ, false, false, nothing))
    U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta)
    @test typeof(U_from_node) <: Float64
    @test isapprox(U_from_node, U_gt; atol=1e-5)

    w = 5.0
    mx = apply_mean_fn(μ_in, mf)
    α = Unimeta.KuuF.L \ Ψ1_trans_approx
    I1 = Ψ0_approx - jdotavx(α,α) # I1 = Ψ0_approx - tr(Unimeta.KuuF \ Ψ2_approx)
    I4 = (
        μ_y^2
        + var(q_out)
        - 2 * μ_y * ( mx + getindex(Ψ1_approx * ( μ_v - Ku_mxu ), 1) )
        + mx^2 
        + tr(R_v * Ψ2_approx)
        + transpose(Ku_mxu) * Ψ2_approx * Ku_mxu
        + 2 * mx * getindex(Ψ1_approx * μ_v, 1) 
        - 2 * mx * getindex(Ψ1_approx * Ku_mxu, 1) 
        - 2 * transpose(μ_v) * Ψ2_approx * Ku_mxu
    )
    U_gt = 0.5 * log(2π) - 0.5 * log(w) + 0.5 * w * (I1 + I4)
    marginals = (Marginal(q_out, false, false, nothing), Marginal(q_in, false, false, nothing), 
                Marginal(q_v, false, false, nothing),Marginal(PointMass(w), false, false, nothing),Marginal(q_θ, false, false, nothing))
    U_from_node = score(AverageEnergy(), UniSGP, Val{(:out, :in, :v, :w, :θ)}(), marginals, Unimeta)
    @test typeof(U_from_node) <: Float64
    @test isapprox(U_from_node, U_gt;atol=1e-5)
end