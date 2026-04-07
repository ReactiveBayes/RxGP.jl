@testitem "node_rule/group_composition/Forward msg T3 = T1 * T2" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 500

    # T1 = small rotation about z
    θ1 = 0.3
    R1 = [cos(θ1) -sin(θ1) 0; sin(θ1) cos(θ1) 0; 0 0 1]
    T1 = Matrix{Float64}(I, 4, 4); T1[1:3,1:3] .= R1; T1[1:3,4] .= [1.0, 0.0, 0.0]

    # T2 = identity
    T2 = Matrix{Float64}(I, 4, 4)

    q_T1 = PoseBelief(T1, 1e-6 * Matrix{Float64}(I, 6, 6))
    q_T2 = PoseBelief(T2, 1e-6 * Matrix{Float64}(I, 6, 6))
    q_T3 = PoseBelief(T1, 0.5 * Matrix{Float64}(I, 6, 6))  # prior on T3

    meta = GroupCompositionMeta(N_samples=N_samp)

    ν_T3 = @call_rule GroupComposition(:T3, Marginalisation) (q_T1=q_T1, q_T2=q_T2, q_T3=q_T3, meta=meta)

    @test ν_T3 isa PoseBelief
    # T3 = T1 * I = T1, so mean should be close to T1
    Δ = logmap(inv(T1) * ν_T3.Tmean)
    @test norm(Δ) < 0.5
end

@testitem "node_rule/group_composition/Backward msg to T1" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 500

    T_id = Matrix{Float64}(I, 4, 4)

    q_T1 = PoseBelief(T_id, 0.5 * Matrix{Float64}(I, 6, 6))
    q_T2 = PoseBelief(T_id, 1e-6 * Matrix{Float64}(I, 6, 6))   # T2 ≈ I
    q_T3 = PoseBelief(T_id, 1e-6 * Matrix{Float64}(I, 6, 6))   # T3 ≈ I

    meta = GroupCompositionMeta(N_samples=N_samp)

    ν_T1 = @call_rule GroupComposition(:T1, Marginalisation) (q_T2=q_T2, q_T3=q_T3, q_T1=q_T1, meta=meta)

    @test ν_T1 isa PoseBelief
    # T1 = T3 * inv(T2) = I * I = I
    Δ = logmap(ν_T1.Tmean)
    @test norm(Δ) < 0.5
end

@testitem "node_rule/group_composition/Backward msg to T2" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 500

    T_id = Matrix{Float64}(I, 4, 4)

    q_T1 = PoseBelief(T_id, 1e-6 * Matrix{Float64}(I, 6, 6))   # T1 ≈ I
    q_T2 = PoseBelief(T_id, 0.5 * Matrix{Float64}(I, 6, 6))
    q_T3 = PoseBelief(T_id, 1e-6 * Matrix{Float64}(I, 6, 6))   # T3 ≈ I

    meta = GroupCompositionMeta(N_samples=N_samp)

    ν_T2 = @call_rule GroupComposition(:T2, Marginalisation) (q_T1=q_T1, q_T3=q_T3, q_T2=q_T2, meta=meta)

    @test ν_T2 isa PoseBelief
    # T2 = inv(T1) * T3 = I * I = I
    Δ = logmap(ν_T2.Tmean)
    @test norm(Δ) < 0.5
end

@testitem "node_rule/group_composition/Average energy" begin
    using RxGP, RxInfer, ReactiveMP, Random, LinearAlgebra, Test, Statistics

    Random.seed!(42)
    N_samp = 1000

    T_id = Matrix{Float64}(I, 4, 4)
    q_T1 = PoseBelief(T_id, 1e-6 * Matrix{Float64}(I, 6, 6))
    q_T2 = PoseBelief(T_id, 1e-6 * Matrix{Float64}(I, 6, 6))
    q_T3 = PoseBelief(T_id, 1e-6 * Matrix{Float64}(I, 6, 6))

    meta = GroupCompositionMeta(N_samples=N_samp)

    E = avg_energy_comp(q_T1, q_T2, q_T3; N=N_samp)
    # T1 * T2 = I ≈ T3, so energy should be very small
    @test E < 0.01
end
