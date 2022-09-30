using ACEcore
using Test

@testset "ACEcore.jl" begin
    @testset "SparseSymmTensor" begin include("test_prodbasis1.jl") end 
end
