
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

function serial_evalpool!(A, basis, bBB) 
   fill!(A, 0.0)
   nX = size(bBB[1], 1)
   for i = 1:nX
      evaluate!(A, basis, ntuple(j -> @view(bBB[j][i, :]), length(bBB)))
   end
   return nothing 
end 


At = zeros(length(spec))
A1 = zeros(length(spec))
A2 = zeros(length(spec))

serial_evalpool!(A1, basis, bBB)
ACEcore.evalpool!(A2, basis, bBB)
A3 = ACEcore.evalpool(basis, bBB)
A4 = ACEcore.test_evalpool(basis, bBB)

@assert A1 ≈ A2 ≈ A3 ≈ A4

@btime serial_evalpool!($A1, $At, $basis, $bBB)
@btime ACEcore.evalpool!($A2, $basis, $bBB)
