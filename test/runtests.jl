using ACEcore
using Test

@testset "ACEcore.jl" begin
    @testset "SparseSymmTensor" begin include("test_prodbasis1.jl") end 
    @testset "Sparse product" begin include("test_1pbasis.jl") end 
end
