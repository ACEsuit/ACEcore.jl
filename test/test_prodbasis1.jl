
using Test
using ACEcore, BenchmarkTools
using ACEcore: SimpleProdBasis, SparseSymmetricProduct, release! 
using Polynomials4ML.Testing: println_slim, print_tf

##

M = 20 
spec = ACEcore.Testing.generate_SO2_spec(5, M)
A = randn(ComplexF64, 2*M+1)

## 

@info("Test consistency of SparseSymmetricProduct with SimpleProdBasis")
basis1 = SimpleProdBasis(spec)
AA1 = basis1(A)

basis2 = SparseSymmetricProduct(spec; T = ComplexF64)
AA2 = basis2(A)

println_slim(@test AA1 ≈ AA2)

## 

@info("Test consistency of serial and batched evaluation")

nX = 32
bA = randn(ComplexF64, nX, 2*M+1)
bAA1 = zeros(ComplexF64, nX, length(spec))
for i = 1:nX
   bAA1[i, :] = basis1(bA[i, :])
end
bAA2 = basis2(bA)

println_slim(@test bAA1 ≈ bAA2)
