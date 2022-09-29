
using Test
using ACEcore, BenchmarkTools
using ACEcore: SimpleProdBasis, SparseSymmetricProduct, release! 
using Polynomials4ML.Testing: println_slim, print_tf

##

M = 30 
spec = ACEcore.Testing.generate_SO2_spec(5, M)

basis1 = SimpleProdBasis(spec)
basis2 = SparseSymmetricProduct(spec; T = Float64)

## ------  Serial performance 

@info("Serial performance")
A = randn(Float64, 2*M+1)

@info("Naive basis")
@btime $basis1($A)

@info("Recursive evaluator")
@btime (AA = $basis2($A); release!(AA))

## ------ batched performance 

@info("Batched performance")

for nX in [4, 8, 16, 32]
   @info("    nX = $nX")
   bA = randn(nX, 2*M+1)
   bAt = collect(bA')
   bAAt = zeros(length(spec), nX)

   basis1 = SimpleProdBasis(spec)
   basis2 = SparseSymmetricProduct(spec; T = Float64)

   function eval_batched!(bAA, basis, bA)
      for i = 1:nX
         AAi = basis((@view bA[:, i]))
         bAA[:, i] = AAi 
         release!(AAi)
      end
      return bAA
   end

   @info("batched evaluation - SimpleProdBasis")
   @btime eval_batched!($bAAt, $basis1, $bAt)

   @info("serial batched evaluation - SparseSymmetricProduct")
   @btime eval_batched!($bAAt, $basis2, $bAt)

   @info("optimized batched evaluation - SparseSymmetricProduct")
   @btime (AA = $basis2($bA); release!(AA))

end