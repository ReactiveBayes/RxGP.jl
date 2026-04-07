@testitem "node_rule/group_inverse_rotation/Forward msg to Y" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 500

    # Pose: 90° rotation about z-axis + translation [1, 0, 0]
    # R maps global→local; Rᵀ maps local→global
    R = [0.0 -1.0 0.0;
         1.0  0.0 0.0;
         0.0  0.0 1.0]
    t = [1.0, 0.0, 0.0]
    T_true = Matrix{Float64}(I, 4, 4)
    T_true[1:3, 1:3] .= R
    T_true[1:3, 4]   .= t

    q_T = PoseBelief(T_true, 1e-6 * Matrix{Float64}(I, 6, 6))
    # Vector [1, 0, 0] in object-local frame
    X_vec = [1.0, 0.0, 0.0]
    q_in = PointMass(X_vec)

    meta = GroupInverseRotationMeta(N_samples=N_samp)

    ν_out = @call_rule GroupInverseRotation(:out, Marginalisation) (q_in=q_in, q_T=q_T, meta=meta)

    # Y = Rᵀ X = [0 1 0; -1 0 0; 0 0 1] * [1,0,0] = [0, -1, 0]
    # (Rᵀ = R⁻¹ for orthogonal R)
    @test typeof(ν_out) <: MultivariateNormalDistributionsFamily
    @test isapprox(mean(ν_out), [0.0, -1.0, 0.0]; atol=0.15)
end

@testitem "node_rule/group_inverse_rotation/Backward msg to X" begin
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
    # Y = [0, -1, 0] in global frame → X = R Y = [0 -1 0; 1 0 0; 0 0 1]*[0,-1,0] = [1, 0, 0]
    Y_vec = [0.0, -1.0, 0.0]
    q_out = PointMass(Y_vec)

    meta = GroupInverseRotationMeta(N_samples=N_samp)

    ν_in = @call_rule GroupInverseRotation(:in, Marginalisation) (q_out=q_out, q_T=q_T, meta=meta)

    @test typeof(ν_in) <: MultivariateNormalDistributionsFamily
    @test isapprox(mean(ν_in), [1.0, 0.0, 0.0]; atol=0.15)
end

@testitem "node_rule/group_inverse_rotation/EP msg to T" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 500

    # Identity pose: R = I  →  Y = Iᵀ X = X
    T_true = Matrix{Float64}(I, 4, 4)

    q_in  = PointMass([1.0, 0.0, 0.0])
    q_out = PointMass([1.0, 0.0, 0.0])   # consistent with R = I
    q_T   = PoseBelief(T_true, 0.5 * Matrix{Float64}(I, 6, 6))

    meta = GroupInverseRotationMeta(N_samples=N_samp)

    ν_T = @call_rule GroupInverseRotation(:T, Marginalisation) (q_out=q_out, q_in=q_in, q_T=q_T, meta=meta)

    @test ν_T isa PoseBelief
    # Recovered rotation should be close to identity
    ξ = logmap(ν_T.Tmean)
    @test norm(ξ[4:6]) < 1.0   # rotational part close to zero
end

@testitem "node_rule/group_inverse_rotation/No translation leakage" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 500

    # Pose with identity rotation but large translation
    T_true = Matrix{Float64}(I, 4, 4)
    T_true[1:3, 4] .= [10.0, 20.0, 30.0]

    q_T = PoseBelief(T_true, 1e-6 * Matrix{Float64}(I, 6, 6))
    X_vec = [1.0, 0.0, 0.0]
    q_in = PointMass(X_vec)

    meta = GroupInverseRotationMeta(N_samples=N_samp)

    ν_out = @call_rule GroupInverseRotation(:out, Marginalisation) (q_in=q_in, q_T=q_T, meta=meta)

    # Y = Iᵀ X = X  (translation should NOT affect the result)
    @test isapprox(mean(ν_out), [1.0, 0.0, 0.0]; atol=0.15)
end
