@testitem "node_rule/pose_belief_prior/Out message is PoseBelief" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test

    Tmean = Matrix{Float64}(I, 4, 4)
    Σ = 2.0 * Matrix{Float64}(I, 6, 6)

    ν_out = @call_rule PoseBeliefPrior(:out, Marginalisation) (
        q_Tmean = PointMass(Tmean),
        q_Σ      = PointMass(Σ)
    )

    @test ν_out isa PoseBelief
    @test ν_out.Tmean == Tmean
    @test ν_out.Σ == Σ
end

@testitem "node_rule/pose_belief_prior/Out message preserves non-identity mean" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test

    # 90° rotation about z
    R = [0.0 -1.0 0.0;
         1.0  0.0 0.0;
         0.0  0.0 1.0]
    Tmean = Matrix{Float64}(I, 4, 4)
    Tmean[1:3, 1:3] .= R
    Tmean[1:3, 4]   .= [1.0, 2.0, 3.0]
    Σ = 0.1 * Matrix{Float64}(I, 6, 6)

    ν_out = @call_rule PoseBeliefPrior(:out, Marginalisation) (
        q_Tmean = PointMass(Tmean),
        q_Σ      = PointMass(Σ)
    )

    @test ν_out.Tmean ≈ Tmean
    @test ν_out.Σ ≈ Σ
end

@testitem "node_rule/pose_belief_prior/Average energy near zero for tight match" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test

    Random.seed!(42)

    Tmean = Matrix{Float64}(I, 4, 4)
    Σ_prior = 1.0 * Matrix{Float64}(I, 6, 6)
    # q_out very tightly centered at the same Tmean → ξ ≈ 0
    q_out = PoseBelief(Tmean, 1e-6 * Matrix{Float64}(I, 6, 6))

    E = @call_rule PoseBeliefPrior(:AverageEnergy, Marginalisation) (
        q_out    = q_out,
        q_Tmean = PointMass(Tmean),
        q_Σ      = PointMass(Σ_prior)
    )

    # The energy should be close to the normalization constant only:
    # ½ log|Σ| + 3 log(2π) ≈ 0 + 3*1.8379 ≈ 5.51
    normalization = 0.5 * logdet(Σ_prior) + 3 * log(2π)
    @test isapprox(E, normalization; atol=0.5)
end
