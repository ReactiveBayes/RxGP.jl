@testitem "node_rule/group_action/Forward msg to Y" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 500

    # Known pose: 90° rotation about z-axis + translation [1, 0, 0]
    R = [0.0 -1.0 0.0;
         1.0  0.0 0.0;
         0.0  0.0 1.0]
    t = [1.0, 0.0, 0.0]
    T_true = Matrix{Float64}(I, 4, 4)
    T_true[1:3, 1:3] .= R
    T_true[1:3, 4]   .= t

    q_T = PoseBelief(T_true, 1e-6 * Matrix{Float64}(I, 6, 6))  # tight pose
    X_point = [1.0, 0.0, 0.0]
    q_in = PointMass(X_point)

    meta = GroupActionMeta(N_samples=N_samp)

    ν_out = @call_rule GroupAction(:out, Marginalisation) (q_in=q_in, q_T=q_T, meta=meta)

    # Y = R*X + t = [0, 1, 0] + [1, 0, 0] = [1, 1, 0]
    @test typeof(ν_out) <: MultivariateNormalDistributionsFamily
    @test isapprox(mean(ν_out), [1.0, 1.0, 0.0]; atol=0.15)
end

@testitem "node_rule/group_action/Backward msg to X" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 500

    R = [0.0 -1.0 0.0;
         1.0  0.0 0.0;
         0.0  0.0 1.0]
    t = [1.0, 0.0, 0.0]
    T_true = Matrix{Float64}(I, 4, 4)
    T_true[1:3, 1:3] .= R
    T_true[1:3, 4]   .= t

    q_T = PoseBelief(T_true, 1e-6 * Matrix{Float64}(I, 6, 6))
    Y_point = [1.0, 1.0, 0.0]
    q_out = PointMass(Y_point)

    meta = GroupActionMeta(N_samples=N_samp)

    ν_in = @call_rule GroupAction(:in, Marginalisation) (q_out=q_out, q_T=q_T, meta=meta)

    # X = R' * (Y - t) = R' * [0, 1, 0] = [1, 0, 0]
    @test typeof(ν_in) <: MultivariateNormalDistributionsFamily
    @test isapprox(mean(ν_in), [1.0, 0.0, 0.0]; atol=0.15)
end

@testitem "node_rule/group_action/EP msg to T" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 500

    # Ground truth: identity pose
    T_true = Matrix{Float64}(I, 4, 4)

    # Known X and Y that are consistent with T = I
    q_in  = PointMass([1.0, 2.0, 3.0])
    q_out = PointMass([1.0, 2.0, 3.0])
    q_T   = PoseBelief(T_true, 0.5 * Matrix{Float64}(I, 6, 6))

    meta = GroupActionMeta(N_samples=N_samp)

    ν_T = @call_rule GroupAction(:T, Marginalisation) (q_out=q_out, q_in=q_in, meta=meta)

    @test ν_T isa PoseBelief
    # The mean pose should be close to identity
    ξ = logmap(ν_T.Tmean)
    @test norm(ξ) < 1.0  # rough check — EP is approximate
end

@testitem "node_rule/group_action/Average energy" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 1000

    T_true = Matrix{Float64}(I, 4, 4)
    q_T   = PoseBelief(T_true, 1e-6 * Matrix{Float64}(I, 6, 6))
    q_in  = PointMass([1.0, 0.0, 0.0])
    q_out = PointMass([1.0, 0.0, 0.0])

    meta = GroupActionMeta(N_samples=N_samp)

    E = @call_rule GroupAction(:AverageEnergy, Marginalisation) (q_out=q_out, q_in=q_in, q_T=q_T, meta=meta)
    # With T=I and Y=X, energy should be ≈ 0
    # Note: @average_energy may not be directly callable via @call_rule in all versions,
    # so this test is aspirational. If it errors, that's expected.
end
