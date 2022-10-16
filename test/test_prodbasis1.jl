
using Test
using ACEcore, BenchmarkTools
using ACEcore: SimpleProdBasis, SparseSymmetricProduct, release!, 
               contract, contract_ed
using Polynomials4ML.Testing: println_slim, print_tf

using ACEbase.Testing: fdtest, dirfdtest

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
using LinearAlgebra: dot

A = randn(2*M+1)
AA2 = real.(basis2(A))

w = randn(Float64, length(spec))'
v1 = contract(w, basis2, A)
v2, g2 = contract_ed(w, basis2, A)
v3 = w * AA2
println_slim(@test v1 ≈ v2 ≈ v3)

for ntest = 1:30 
   U = randn(length(A))
   F = t -> contract(w, basis2, A + t * U)
   dF = t -> real(dot(contract_ed(w, basis2, A + t * U)[2], U))
   print_tf(@test fdtest(F, dF, 0.0; verbose=false))
end
println() 

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

## 

@info("Test batched pullback")

for ntest = 1:10 
   nX = 32
   bA = randn(nX, 2*M+1)
   bAA = zeros(nX, length(basis2.dag))
   ACEcore.evaluate_dag!(bAA, basis2.dag, bA)
   b∂A = zero(bA)
   b∂AA = randn(nX, length(basis2.dag))
   ACEcore.pullback_evaluate_dag!(b∂A, copy(b∂AA), basis2, bAA)

   b∂A1 = zero(bA)
   for j = 1:nX 
      ACEcore.pullback_evaluate_dag!( (@view b∂A1[j, :]), 
                        b∂AA[j, :], basis2, bAA[j, :])
   end 

   print_tf(@test b∂A1 ≈ b∂A)
end
