
using ACEcore, BenchmarkTools
using ACEcore:  PooledSparseProduct, evaluate , evaluate!, 
                evalpool, evalpool!, 
                acquire!, release! 

##

@info(" -----------  PooledSparseProduct -------------- ")

N1 = 10 
N2 = 20 
N3 = 50 
spec = sort([ (rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:3_000 ])

basis = PooledSparseProduct(spec)

nX = 64 
bBB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )

A1 = zeros(length(spec))

evalpool!(A1, basis, bBB)
A2 = evalpool(basis, bBB)

@assert A1 ≈ A2

##

function runn!(N, A, basis, bBB)
   for _=1:N
      evalpool!(A, basis, bBB)
   end
end

function runn(N, basis, bBB)
   for _=1:N
      evalpool(basis, bBB)
   end
end

@info("in-place")
@btime runn!(1_000, $A1, $basis, $bBB)
@info("allocating")
@btime runn(1_000, $basis, $bBB)
@info("in-place again")
@btime runn!(1_000, $A1, $basis, $bBB)


##

@info(" -----------  SparseSymmProd -------------- ")
using ACEcore: SparseSymmProd

M = 30 
spec = ACEcore.Testing.generate_SO2_spec(5, M)
basis = SparseSymmProd(spec; T = Float64)
basis = basis.dag
bA = randn(nX, 2*M+1)
bAA = zeros(nX, length(basis))

evaluate(basis, bA)
evaluate!(bAA, basis, bA)

##

@info("in-place")
@btime evaluate!($bAA, $basis, $bA)
@info("Allocating")
@btime evaluate($basis, $bA)
@info("With cache")
@btime (AA = evaluate($basis, $bA); release!(AA))
@info("in-place again")
@btime evaluate!($bAA, $basis, $bA)

