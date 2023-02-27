
using Test
using ACEcore, BenchmarkTools
using ACEcore: SimpleProdBasis, release!, SparseSymmProd
using Polynomials4ML.Testing: println_slim, print_tf

using ACEbase.Testing: fdtest, dirfdtest

##

M = 20 
spec = ACEcore.Testing.generate_SO2_spec(5, M)
A = randn(ComplexF64, 2*M+1)

## 

@info("Test consistency of SparseSymmetricProduct with SimpleProdBasis")
AA1 = basis1(A)
basis1 = SimpleProdBasis(spec)

basis2 = SparseSymmProd(spec; T = ComplexF64)
AA2 = basis2(A)

spec_ = ACEcore.reconstruct_spec(basis2.dag)[basis2.proj]
println_slim(@test spec_ == spec)
println_slim(@test AA1 ≈ AA2)

##

@info("Test with a constant")
spec_c = [ [Int[],]; spec]
basis1_c = SimpleProdBasis(spec_c)
basis2_c = SparseSymmProd(spec_c; T = ComplexF64)

spec_c_ = ACEcore.reconstruct_spec(basis2_c.dag)[basis2_c.proj]
println_slim(@test spec_c_ == spec_c)

AA1_c = basis1_c(A)
println_slim(@test AA1 ≈ AA1_c[2:end])
println_slim(@test AA1_c[1] ≈ 1.0)

AA2_c = basis2_c(A)
println_slim(@test AA2_c[1] ≈ 1.0)
println_slim(@test AA2_c ≈ AA1_c)


##   TODO: MOVE TO A LINEAR MODEL PROTOTYPE IMPLEMENTATION 

# @info("Test gradient of SparseSymmetricProduct") 

# using ACEcore: contract, contract_ed
# using LinearAlgebra: dot

# A = randn(2*M+1)
# AA2 = real.(basis2(A))

# w = randn(Float64, length(spec))'
# v1 = contract(w, basis2, A)
# v2, g2 = contract_ed(w, basis2, A)
# v3 = w * AA2
# println_slim(@test v1 ≈ v2 ≈ v3)

# for ntest = 1:30 
#    U = randn(length(A))
#    F = t -> contract(w, basis2, A + t * U)
#    dF = t -> real(dot(contract_ed(w, basis2, A + t * U)[2], U))
#    print_tf(@test fdtest(F, dF, 0.0; verbose=false))
# end
# println() 

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

@info("Test batched pullback DAG")

for ntest = 1:20 
   local nX, bA 
   nX = 32
   bA = randn(nX, 2*M+1)
   bAA = zeros(nX, length(basis2.dag))
   ACEcore.evaluate!(bAA, basis2.dag, bA)
   b∂A = zero(bA)
   b∂AA = randn(nX, length(basis2.dag))
   ACEcore.pullback_arg!(b∂A, copy(b∂AA), basis2.dag, bAA)

   b∂A1 = zero(bA)
   for j = 1:nX 
      ACEcore.pullback_arg!( (@view b∂A1[j, :]), 
                        b∂AA[j, :], basis2.dag, bAA[j, :])
   end 

   print_tf(@test b∂A1 ≈ b∂A)
end
println() 


##

# nX = 32
# bA = randn(nX, 2*M+1)
# bAA = zeros(nX, length(basis2.dag))
# ACEcore.evaluate!(bAA, basis2.dag, bA)
# b∂A = zero(bA)
# b∂AA = randn(nX, length(basis2.dag))
# ACEcore.pullback_arg!(b∂A, copy(b∂AA), basis2.dag, bAA)


# @profview let b∂A=b∂A, b∂AA=b∂AA, dag = basis2.dag, bAA = bAA
#    for n = 1:100_000
#       ACEcore.pullback_arg!(b∂A, b∂AA, basis2.dag, bAA)
#    end
# end
