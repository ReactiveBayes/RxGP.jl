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
        meta = get_UniSGPMeta(D; method=grad_default_method, mean_fn=test_mean_fn, kernel=kernel, kernel_spec=kernel_spec, mode=:AN, independent_SE_lengthscales=true, Xu=Xu, θ=θ_val)
        return (; method=grad_default_method, D, Xu, Nu=length(Xu), kernel, θ_val, meta, q_x, q_out, q_v, q_Wg, q_θ=PointMass(θ_val), rng)
    end
end

@testitem "helper_functions/approximate_kernel/Test predict_GP_values" setup=[setup_snippet] begin
    ctx = test_fixture(;D=1)
    value_pred_means, value_pred_var = predict_GP_values(; m_in=[[1.3], [2.5], [-213]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_value_means_1d = [1.426392254114819, 1.5048343203596062, -213.0]
    expected_value_var_1d = [12.295892216916247, 1.7208706079583371, 0.95]
    @test isapprox(value_pred_means, expected_value_means_1d; atol=1e-6)
    @test isapprox(value_pred_var, expected_value_var_1d; atol=1e-6)
    
    ctx = test_fixture(;D=2)
    value_pred_means, value_pred_var = predict_GP_values(; m_in=[[1.3, 0.5], [2.5, -1.2], [-213, 4.4]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_value_means_2d = [1.2984664489437887, 1.251521520847584, -208.6]
    expected_value_var_2d = [2.8484427880442604, 1.0067644853938489, 0.95]
    @test isapprox(value_pred_means, expected_value_means_2d; atol=1e-6)
    @test isapprox(value_pred_var, expected_value_var_2d; atol=1e-6)
    
    ctx = test_fixture(;D=3)
    value_pred_means, value_pred_var = predict_GP_values(; m_in=[[1.3, 0.5, -0.7], [2.5, -1.2, 3.3], [-213, 4.4, 0.1]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_value_means_3d = [0.5818501472216481, 4.582220642228733, -208.5]
    expected_value_var_3d = [2.906792054169222, 0.9501651809610067, 0.95]
    @test isapprox(value_pred_means, expected_value_means_3d; atol=1e-6)
    @test isapprox(value_pred_var, expected_value_var_3d; atol=1e-6)

end



@testitem "helper_functions/approximate_kernel/Test predict_GP_gradients" setup=[setup_snippet] begin
    ctx = test_fixture(;D=1)
    grad_pred_means, grad_pred_cov = predict_GP_gradients(; m_in=[[1.3], [2.5], [-213]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_grad_means_1d = [[-0.673068277833309], [1.202928520977025], [1.0]]
    expected_grad_cov_1d = [[2.39163541314601;;], [3.402633523216112;;], [0.8616780045351473;;]]
    @test all(isapprox.(grad_pred_means, expected_grad_means_1d; atol=1e-6))
    @test all(isapprox.(grad_pred_cov, expected_grad_cov_1d; atol=1e-6))

    ctx = test_fixture(;D=2)
    grad_pred_means, grad_pred_cov = predict_GP_gradients(; m_in=[[1.3, 0.5], [2.5, -1.2], [-213, 4.4]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_grad_means_2d = [[0.032305775592376906, 0.284284625145349], [0.8139161086436225, 0.7321200422666407], [1.0, 1.0]]
    expected_grad_cov_2d = [[2.025067843061719 1.0285131398444358; 1.0285131398444343 1.8068230232745943], [1.1671355815266986 -0.10958746306617301; -0.10958746306617298 0.8862106194320802], [0.95 0.0; 0.0 0.8616780045351473]]
    @test all(isapprox.(grad_pred_means, expected_grad_means_2d; atol=1e-6))
    @test all(isapprox.(grad_pred_cov, expected_grad_cov_2d; atol=1e-6))
    
    ctx = test_fixture(;D=3)
    grad_pred_means, grad_pred_cov = predict_GP_gradients(; m_in=[[1.3, 0.5, -0.7], [2.5, -1.2, 3.3], [-213, 4.4, 0.1]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_grad_means_3d = [[1.3127851181443715, 1.0755124280131763, 0.6538097995285279], [1.042091553446812, 0.9764420644109422, 1.029424758224543], [1.0, 1.0, 1.0]]
    expected_grad_cov_3d = [[1.4857306425980172 -0.01497847595330472 0.008369286021698464; -0.014978475953304721 0.9182837536426514 0.012592864200764442; 0.008369286021698482 0.012592864200764445 0.9162844471777095], [0.9834172560632981 -0.0005248641121455148 0.0006539768987931248; -0.0005248641121455149 0.919400538944933 -0.00036480593475292914; 0.0006539768987931249 -0.0003648059347529292 0.8621325975032718], [0.9824762999138179 0.0 0.0; 0.0 0.9191077667293739 0.0; 0.0 0.0 0.8616780045351473]]
    @test all(isapprox.(grad_pred_means, expected_grad_means_3d; atol=1e-6))
    @test all(isapprox.(grad_pred_cov, expected_grad_cov_3d; atol=1e-6))

end



@testitem "helper_functions/approximate_kernel/Test predict_GP_joints" setup=[setup_snippet] begin
    ctx = test_fixture(;D=1)
    joint_pred_means, joint_pred_grads = predict_GP_joints(; m_in=[[1.3], [2.5], [-213]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_joint_means_1d = [1.426392254114819 -0.673068277833309; 1.5048343203596062 1.202928520977025; -213.0 1.0]
    expected_joint_grads_1d = [12.295892216916247 -9.425995231080496; 1.7208706079583371 -3.3011008425616613; 0.95 0.0;;; -9.425995231080496 2.39163541314601; -3.3011008425616613 3.402633523216112; 0.0 0.8616780045351473]
    @test isapprox(joint_pred_means, expected_joint_means_1d; atol=1e-6)
    @test isapprox(joint_pred_grads, expected_joint_grads_1d; atol=1e-6)
    
    ctx = test_fixture(;D=2)
    joint_pred_means, joint_pred_grads = predict_GP_joints(; m_in=[[1.3, 0.5], [2.5, -1.2], [-213, 4.4]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_joint_means_2d = [1.2984664489437887 0.032305775592376906 0.284284625145349; 1.251521520847584 0.8139161086436225 0.7321200422666407; -208.6 1.0 1.0]
    expected_joint_grads_2d = [2.8484427880442604 -3.4628430482407735 -3.1414526503728752; 1.0067644853938489 -0.2319717579533537 0.09214016820423497; 0.95 0.0 0.0;;; -3.4628430482407735 2.025067843061719 1.0285131398444343; -0.2319717579533537 1.1671355815266986 -0.10958746306617298; 0.0 0.95 0.0;;; -3.1414526503728752 1.0285131398444358 1.8068230232745943; 0.09214016820423497 -0.10958746306617301 0.8862106194320802; 0.0 0.0 0.8616780045351473]
    @test isapprox(joint_pred_means, expected_joint_means_2d; atol=1e-6)
    @test isapprox(joint_pred_grads, expected_joint_grads_2d; atol=1e-6)
    
    ctx = test_fixture(;D=3)
    joint_pred_means, joint_pred_grads = predict_GP_joints(; m_in=[[1.3, 0.5, -0.7], [2.5, -1.2, 3.3], [-213, 4.4, 0.1]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_joint_means_3d = [0.5818501472216481 1.3127851181443715 1.0755124280131763 0.6538097995285279; 4.582220642228733 1.042091553446812 0.9764420644109422 1.029424758224543; -208.5 1.0 1.0 1.0]
    expected_joint_grads_3d = [2.906792054169222 -1.940759088822373 0.0424726861077448 -0.20872778346473422; 0.9501651809610067 -0.0007885221673454607 0.0004398318803321496 -0.0005480470967678551; 0.95 0.0 0.0 0.0;;; -1.940759088822373 1.4857306425980172 -0.014978475953304721 0.008369286021698482; -0.0007885221673454607 0.9834172560632981 -0.0005248641121455149 0.0006539768987931249; 0.0 0.9824762999138179 0.0 0.0;;; 0.0424726861077448 -0.01497847595330472 0.9182837536426514 0.012592864200764445; 0.0004398318803321496 -0.0005248641121455148 0.919400538944933 -0.0003648059347529292; 0.0 0.0 0.9191077667293739 0.0;;; -0.20872778346473422 0.008369286021698464 0.012592864200764442 0.9162844471777095; -0.0005480470967678551 0.0006539768987931248 -0.00036480593475292914 0.8621325975032718; 0.0 0.0 0.0 0.8616780045351473]
    @test isapprox(joint_pred_means, expected_joint_means_3d; atol=1e-6)
    @test isapprox(joint_pred_grads, expected_joint_grads_3d; atol=1e-6)

end