
using ACEcore, BenchmarkTools
using ACEcore:  PooledSparseProduct, test_evaluate, evaluate , evaluate!

function evalpool_multi!(A, bA, BBB)
   for i = 1:length(BBB)
      ACEcore.evalpool!(@view(A[:, i]), bA, BBB[i])
   end
   return nothing 
end

function pb_evalpool_multi!(∂BBB, ∂A, bA, BBB)
   for i = 1:length(BBB)
      ACEcore._pullback_evalpool!(∂BBB[i], @view(∂A[i, :]), bA, BBB[i])
   end
   return nothing 
end

##


N1 = 10 
N2 = 20 
N3 = 50 
spec = sort([ (rand(1:N1), rand(1:N2), rand(1:N3)) for i = 1:100 ])
basis = PooledSparseProduct(spec)

##

@info("Test batched evaluation")
nnX = [30, 33, 25, 13] 
sum_nnX = sum(nnX)
bBB1 = [ ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )  for nX in nnX]
bBB2 = tuple( [ vcat([bBB1[i][j] for i = 1:length(nnX)]...) for j=1:3 ]... )
target = vcat([ fill(i, nnX[i]) for i = 1:length(nnX)]...)

A1 = zeros(length(spec), length(nnX))
A2 = zeros(length(nnX), length(spec))

evalpool_multi!(A1, basis, bBB1)
ACEcore.evalpool!(A2, basis, bBB2, target)

@show A1' ≈ A2

## 

@info("sequential")
@btime evalpool_multi!($A1, $basis, $bBB1)
@info("with target")
@btime ACEcore.evalpool!($A2, $basis, $bBB2)

## 

# @profview let A2 = A2, basis = basis, bBB2 = bBB2
#    for _ = 1:1_000_000 
#       ACEcore.evalpool!(A2, basis, bBB2)
#    end
# end

##

∂A = randn(size(A2))
∂BB1 = deepcopy(bBB1)
∂BB2 = deepcopy(bBB2)

pb_evalpool_multi!(∂BB1, ∂A, basis, bBB1)
ACEcore._pullback_evalpool!(∂BB2, ∂A, basis, bBB2, target)

_∂BB1_ = tuple( [ vcat([∂BB1[i][j] for i = 1:length(nnX)]...) for j=1:3 ]... )
all(_∂BB1_ .≈ ∂BB2)

##

@info("sequential pb")
@btime pb_evalpool_multi!($∂BB1, $∂A, $basis, $bBB1)

@info("pb with target")
@btime ACEcore._pullback_evalpool!($∂BB2, $∂A, $basis, $bBB2, $target)

## 

@profview let ∂BB2 = ∂BB2, ∂A = ∂A, basis = basis, bBB2 = bBB2, target = target
   for _ = 1:400_000 
      ACEcore._pullback_evalpool!(∂BB2, ∂A, basis, bBB2, target)
   end
end
