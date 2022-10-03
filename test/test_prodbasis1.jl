
using Test
using ACEcore, BenchmarkTools
using ACEcore: SimpleProdBasis, SparseSymmetricProduct, release!, 
               contract, contract_ed
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

@info("Test gradient of SparseSymmetricProduct") 

using ACEcore: contract, contract_ed
w = randn(Float64, length(spec))'
v1 = contract(w, basis2, A)
v2, g2 = contract_ed(w, basis2, A)
v3 = w * AA2
println_slim(@test v1 ≈ v2 ≈ v3)



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

