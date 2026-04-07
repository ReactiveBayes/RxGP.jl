@testsnippet pred_snippet begin
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

    function test_fixture(; D=1, operator=:fn)
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
        meta = get_UniSGPMeta(D; method=grad_default_method, mean_fn=test_mean_fn, kernel=kernel, kernel_spec=kernel_spec, mode=:AN, operator=operator, independent_SE_lengthscales=true, Xu=Xu, θ=θ_val)
        return (; method=grad_default_method, D, Xu, Nu=length(Xu), kernel, θ_val, meta, q_x, q_out, q_v, q_Wg, q_θ=PointMass(θ_val), rng)
    end
end

@testitem "helper_functions/univariate_prediction_fns/Test predict_GP (fn operator)" setup=[pred_snippet] begin
    # operator=:fn → P=1: means[i] is 1-element vector, covs[i] is 1×1 matrix.
    ctx = test_fixture(; D=1, operator=:fn)
    pred_means, pred_covs = predict_GP(; m_in=[[1.3], [2.5], [-213]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_means_1d = [1.426392254114819, 1.5048343203596062, -213.0]
    expected_var_1d   = [12.295892216916247, 1.7208706079583371, 0.95]
    @test isapprox([m[1] for m in pred_means], expected_means_1d; atol=1e-6)
    @test isapprox([c[1,1] for c in pred_covs], expected_var_1d; atol=1e-6)

    ctx = test_fixture(; D=2, operator=:fn)
    pred_means, pred_covs = predict_GP(; m_in=[[1.3, 0.5], [2.5, -1.2], [-213, 4.4]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_means_2d = [1.2984664489437887, 1.251521520847584, -208.6]
    expected_var_2d   = [2.8484427880442604, 1.0067644853938489, 0.95]
    @test isapprox([m[1] for m in pred_means], expected_means_2d; atol=1e-6)
    @test isapprox([c[1,1] for c in pred_covs], expected_var_2d; atol=1e-6)

    ctx = test_fixture(; D=3, operator=:fn)
    pred_means, pred_covs = predict_GP(; m_in=[[1.3, 0.5, -0.7], [2.5, -1.2, 3.3], [-213, 4.4, 0.1]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_means_3d = [0.5818501472216481, 4.582220642228733, -208.5]
    expected_var_3d   = [2.906792054169222, 0.9501651809610067, 0.95]
    @test isapprox([m[1] for m in pred_means], expected_means_3d; atol=1e-6)
    @test isapprox([c[1,1] for c in pred_covs], expected_var_3d; atol=1e-6)
end



@testitem "helper_functions/univariate_prediction_fns/Test predict_GP (grad operator)" setup=[pred_snippet] begin
    # operator=:grad → P=D: means[i] is D-vector, covs[i] is D×D matrix.
    ctx = test_fixture(; D=1, operator=:grad)
    pred_means, pred_covs = predict_GP(; m_in=[[1.3], [2.5], [-213]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_grad_means_1d = [[-0.673068277833309], [1.202928520977025], [1.0]]
    expected_grad_cov_1d   = [[2.39163541314601;;], [3.402633523216112;;], [0.8616780045351473;;]]
    @test all(isapprox.(pred_means, expected_grad_means_1d; atol=1e-6))
    @test all(isapprox.(pred_covs, expected_grad_cov_1d; atol=1e-6))

    ctx = test_fixture(; D=2, operator=:grad)
    pred_means, pred_covs = predict_GP(; m_in=[[1.3, 0.5], [2.5, -1.2], [-213, 4.4]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_grad_means_2d = [[0.032305775592376906, 0.284284625145349], [0.8139161086436225, 0.7321200422666407], [1.0, 1.0]]
    expected_grad_cov_2d   = [[2.025067843061719 1.0285131398444358; 1.0285131398444343 1.8068230232745943], [1.1671355815266986 -0.10958746306617301; -0.10958746306617298 0.8862106194320802], [0.95 0.0; 0.0 0.8616780045351473]]
    @test all(isapprox.(pred_means, expected_grad_means_2d; atol=1e-6))
    @test all(isapprox.(pred_covs, expected_grad_cov_2d; atol=1e-6))

    ctx = test_fixture(; D=3, operator=:grad)
    pred_means, pred_covs = predict_GP(; m_in=[[1.3, 0.5, -0.7], [2.5, -1.2, 3.3], [-213, 4.4, 0.1]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_grad_means_3d = [[1.3127851181443715, 1.0755124280131763, 0.6538097995285279], [1.042091553446812, 0.9764420644109422, 1.029424758224543], [1.0, 1.0, 1.0]]
    expected_grad_cov_3d   = [[1.4857306425980172 -0.01497847595330472 0.008369286021698464; -0.014978475953304721 0.9182837536426514 0.012592864200764442; 0.008369286021698482 0.012592864200764445 0.9162844471777095], [0.9834172560632981 -0.0005248641121455148 0.0006539768987931248; -0.0005248641121455149 0.919400538944933 -0.00036480593475292914; 0.0006539768987931249 -0.0003648059347529292 0.8621325975032718], [0.9824762999138179 0.0 0.0; 0.0 0.9191077667293739 0.0; 0.0 0.0 0.8616780045351473]]
    @test all(isapprox.(pred_means, expected_grad_means_3d; atol=1e-6))
    @test all(isapprox.(pred_covs, expected_grad_cov_3d; atol=1e-6))
end



@testitem "helper_functions/univariate_prediction_fns/Test predict_GP (joint_fn_grad operator)" setup=[pred_snippet] begin
    # operator=:joint_fn_grad → P=1+D: means[i] is (1+D)-vector, covs[i] is (1+D)×(1+D) matrix.
    # Expected means matrices: row i = [fn_val, grad_val_1, ..., grad_val_D] for input i.
    # Expected covs arrays: A[i,:,:] is the (1+D)×(1+D) covariance for input i.
    ctx = test_fixture(; D=1, operator=:joint_fn_grad)
    pred_means, pred_covs = predict_GP(; m_in=[[1.3], [2.5], [-213]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_means_1d = [1.426392254114819 -0.673068277833309; 1.5048343203596062 1.202928520977025; -213.0 1.0]
    expected_covs_1d  = [[12.295892214308957 -4.71299761482039; -4.71299761482039 2.391635412806453], [1.7208706056690555 -1.6505504180975399; -1.6505504180975399 3.402633524100966], [0.9500000000000002 0.0; 0.0 0.8616780045351471]]
    for i in 1:3
        @test isapprox(pred_means[i], expected_means_1d[i,:]; atol=1e-6)
        @test isapprox(Matrix(pred_covs[i]), expected_covs_1d[i]; atol=1e-2)
    end

    ctx = test_fixture(; D=2, operator=:joint_fn_grad)
    pred_means, pred_covs = predict_GP(; m_in=[[1.3, 0.5], [2.5, -1.2], [-213, 4.4]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_means_2d = [1.2984664489437887 0.032305775592376906 0.284284625145349; 1.251521520847584 0.8139161086436225 0.7321200422666407; -208.6 1.0 1.0]
    expected_covs_2d  = [[2.848442788044262 -1.731421524120386 -1.5707263251864372; -1.731421524120386 2.025067843061719 1.0285131398444363; -1.5707263251864372 1.0285131398444363 1.8068230232745957], [1.006764485393849 -0.115985878976677 0.04607008410211748; -0.115985878976677 1.1671355815266988 -0.10958746306617302; 0.04607008410211748 -0.10958746306617302 0.88621061943208], [0.9500000000000002 0.0 0.0; 0.0 0.9499999999999997 0.0; 0.0 0.0 0.8616780045351471]]
    for i in 1:3
        @test isapprox(pred_means[i], expected_means_2d[i,:]; atol=1e-6)
        @test isapprox(Matrix(pred_covs[i]), expected_covs_2d[i]; atol=1e-2)
    end

    ctx = test_fixture(; D=3, operator=:joint_fn_grad)
    pred_means, pred_covs = predict_GP(; m_in=[[1.3, 0.5, -0.7], [2.5, -1.2, 3.3], [-213, 4.4, 0.1]], q_v=ctx.q_v, q_θ=ctx.q_θ, meta=ctx.meta)
    expected_means_3d = [0.5818501472216481 1.3127851181443715 1.0755124280131763 0.6538097995285279; 4.582220642228733 1.042091553446812 0.9764420644109422 1.029424758224543; -208.5 1.0 1.0 1.0]
    expected_covs_3d  = [[2.9067920541692227 -0.9703795444111862 0.021236343053872633 -0.10436389173236772; -0.9703795444111862 1.4857306425980168 -0.014978475953304836 0.00836928602169859; 0.021236343053872633 -0.014978475953304836 0.9182837536426512 0.01259286420076445; -0.10436389173236772 0.00836928602169859 0.01259286420076445 0.9162844471777096], [0.9501651809610069 -0.0003942610836727317 0.00021991594016607563 -0.00027402354838392863; -0.0003942610836727317 0.9834172560632978 -0.0005248641121455165 0.0006539768987931269; 0.00021991594016607563 -0.0005248641121455165 0.9194005389449329 -0.00036480593475293033; -0.00027402354838392863 0.0006539768987931269 -0.00036480593475293033 0.8621325975032715], [0.9500000000000002 0.0 0.0 0.0; 0.0 0.9824762999138176 0.0 0.0; 0.0 0.0 0.9191077667293738 0.0; 0.0 0.0 0.0 0.8616780045351471]]
    for i in 1:3
        @test isapprox(pred_means[i], expected_means_3d[i,:]; atol=1e-6)
        @test isapprox(Matrix(pred_covs[i]), expected_covs_3d[i]; atol=1e-2)
    end
end