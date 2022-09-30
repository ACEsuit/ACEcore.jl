
using ACEcore, BenchmarkTools
using ACEcore:  PooledSparseProduct, test_evaluate, evaluate , evaluate!

N1 = 10 
N2 = 20 
N3 = 50 
spec = sort([ (rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:100 ])

basis = PooledSparseProduct(spec)


@info("Test batched evaluation")
nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )

function serial_prod_and_pool!(A, At, basis, bBB) 
   fill!(A, 0.0)
   nX = size(bBB[1], 1)
   for i = 1:nX
      evaluate!(At, basis, ntuple(j -> @view(bBB[j][i, :]), length(bBB)))
      A += At
   end
   return A
end 

At = zeros(length(spec))
A1 = zeros(length(spec))
A2 = zeros(length(spec))

A1 = serial_prod_and_pool!(A1, At, basis, bBB)
A2 = ACEcore.prod_and_pool!(A2, basis, bBB)
A1 ≈ A2

@btime serial_prod_and_pool!($A1, $At, $basis, $bBB)
@btime ACEcore.prod_and_pool!($A2, $basis, $bBB)

