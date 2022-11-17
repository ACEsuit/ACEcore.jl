
using ACEcore, BenchmarkTools
using ACEcore:  PooledSparseProduct, test_evaluate, evaluate , evaluate!

struct ZeroNoEffect end 
Base.size(::ZeroNoEffect, ::Integer) = Inf
Base.setindex!(A::ZeroNoEffect, args...) = nothing
Base.getindex(A::ZeroNoEffect, args...) = Bool(0)


##


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

@btime serial_evalpool!($A1, $basis, $bBB)
@btime ACEcore.evalpool!($A2, $basis, $bBB)

##

@profview let A2=A2, basis=basis, bBB=bBB
   for _=1:3_000_000
      ACEcore.evalpool!(A2, basis, bBB)
   end
end

##

@profview let A2=A2, basis=basis, bBB=bBB
   for _=1:3_000_000
      ACEcore.evalpool(basis, bBB)
   end
end



## 

∂A = randn(size(A1))
val, pb = ACEcore._rrule_evalpool(basis, bBB)
∂BB = pb(∂A)

@info("timing pullback")
@btime begin 
   val, pb = ACEcore._rrule_evalpool($basis, $bBB)
   ∂BB = pb($∂A)
end

##

@info("timing in-place pullback")
display( @benchmark ACEcore._pullback_evalpool!($∂BB, $∂A, $basis, $bBB) )

##

# cost with ZeroNoEffect

∂BB1 = (∂BB[1], ∂BB[2], ZeroNoEffect())
ACEcore._pullback_evalpool!(∂BB1, ∂A, basis, bBB)
display( @benchmark  ACEcore._pullback_evalpool!($∂BB1, $∂A, $basis, $bBB) )
@btime  ACEcore._pullback_evalpool!($∂BB1, $∂A, $basis, $bBB)

##



using Enzyme, LinearAlgebra

∂BBe = map(zero, bBB)
Ae = zero(A1)

autodiff(Reverse, ACEcore.evalpool!, Const, Duplicated(Ae, ∂A), Const(basis), 
         Duplicated(bBB, ∂BBe))

@show Ae ≈ val         
@show all(∂BBe .≈ ∂BB)

@btime begin 
   autodiff(Reverse, ACEcore.evalpool!, Const, Duplicated($Ae, $∂A), Const($basis), 
                Duplicated($bBB, $∂BBe))
end


# ##

# using Enzyme

# BB = ( randn(nX, N1), randn(nX, N2), randn(nX, N3) )
# ∂BB = map(zero, BB)

# A = zeros(100)
# dA = ones(100)

# autodiff(Reverse, ACEcore.evalpool!, Const, Duplicated(A, dA), Const(basis), Duplicated(BB, ∂BB))
