
using ACEcore, BenchmarkTools, Test
using ACEcore:  PooledSparseProduct, test_evaluate, evaluate , evaluate!, 
                evalpool
using Polynomials4ML.Testing: println_slim

N1 = 10 
N2 = 20 
N3 = 50 

B1 = randn(N1)
B2 = randn(N2)
B3 = randn(N3)

spec = sort([ (rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:100 ])

basis = PooledSparseProduct(spec)

## 

@info("Test serial evaluation")

BB = (B1, B2, B3)

A1 = test_evaluate(basis, BB)
A2 = evaluate(basis, BB)

println_slim(@test A1 ≈ A2 )

## 

@info("Test batched evaluation")
nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )

# using the naive evaluation code 
bA1 = sum(  test_evaluate(basis, ntuple(j -> bBB[j][i, :], length(bBB)))
            for i = 1:nX )

bA2 = evalpool(basis, bBB)

println_slim(@test bA1 ≈ bA2 )


##

using StaticArrays

prodgrad = ACEcore._prod_grad

for 

b = rand(SVector{3, Float64})
v, g = prodadj(b)

v ≈ prod(b)
g ≈ [ b[2] * b[3], b[1] * b[3], b[1] * b[2] ]

b = rand(SVector{2, Float64})
v, g = prodadj(b)
v ≈ prod(b)
g ≈ [ b[2], b[1] ]


b = rand(SVector{1, Float64})
v, g = prodadj(b)
v ≈ b[1]
g ≈ [ one(b[1]) ]

##