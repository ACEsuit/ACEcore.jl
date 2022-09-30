
using ACEcore, BenchmarkTools
using ACEcore:  PooledSparseProduct, test_evaluate, evaluate , evaluate!

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

A1 ≈ A2 

## 

@info("Test batched evaluation")
nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )

# using the naive evaluation code 
bA1 = sum(  test_evaluate(basis, ntuple(j -> bBB[j][i, :], length(bBB)))
            for i = 1:nX )


bA2 = zeros(length(spec))            
bA2 = ACEcore.prod_and_pool!(bA2, basis, bBB)
bA2_ = ACEcore.prod_and_pool3!(bA2, basis, bBB)

bA1 ≈ bA2 ≈ bA2_

##

@btime evaluate!($A2, $basis, $BB)

@btime ACEcore.prod_and_pool!($bA2, $basis, $bBB)

@btime ACEcore.prod_and_pool3!($bA2, $basis, $bBB)

