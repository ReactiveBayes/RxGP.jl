using RxGP
using Random
using LinearAlgebra
using Test
using TestItemRunner

@testitem "helper_functions/other_functions/TestOtherFunction" begin
    using RxGP, Random, LinearAlgebra, Test
    A = rand(4,4)
    blk_A = [A[1:2,1:2], A[3:4,1:2], A[1:2,3:4],A[3:4,3:4]]
    a = rand(4)
    b = rand(4)

    @test jdotavx(a,b) ≈ dot(a,b)
    blk_matrix = create_blockmatrix(A,2,2)
    for i in eachindex(blk_matrix)
        @test blk_matrix[i] == blk_A[i]
    end
end