

using ACE1, ACEcore, BenchmarkTools
using ACE1: PolyTransform, transformed_jacobi

##
# create a real potential 
# tune the degree to get different size graphs, I like the following: 
#
# maxdeg  | basis size 
#   12    |  ≈ 7k
#   15    |  ≈ 33k
#   18    |  ≈ 135k 
#   21    |  ≈ 466k 

@info("Basic test of PIPotential construction and evaluation")
maxdeg = 10
order = 3
r0 = 1.0
rcut = 3.0
trans = PolyTransform(1, r0)
Pr = transformed_jacobi(maxdeg, trans, rcut; pcut = 2)
D = ACE1.SparsePSHDegree()
P1 = ACE1.BasicPSH1pBasis(Pr; species = :X, D = D)
basis = ACE1.PIBasis(P1, order, D, maxdeg)
@show length(basis)

## 

spec = ACE1.get_basis_spec(basis, 1)
inv_spec = Dict{Any, Int}() 
for (i, bb) in enumerate(spec)
   inv_spec[bb] = i
end

_refl(b::ACE1.RPI.PSH1pBasisFcn) = ACE1.RPI.PSH1pBasisFcn(b.n, b.l, -b.m, b.z)
_refl(bb::ACE1.PIBasisFcn) = ACE1.PIBasisFcn(bb.z0, _refl.(bb.oneps), bb.top)

spec_refl = [_refl(bb) for bb in spec ]

mirrors = [ haskey(inv_spec, bb) ? inv_spec[bb] : missing for bb in spec_refl ]
count(ismissing, mirrors) + count([ i === mirrors[i] for i = 1:length(spec) ])

##

@show length(basis)

## 
# convert to the new format 

orders = basis.inner[1].orders
iAA2iA = basis.inner[1].iAA2iA
new_spec = [ iAA2iA[i, 1:orders[i]][:] for i = 1:length(orders) ]

t2iAA = Dict{Vector{Int}, Int}()
for (iAA, bb) in enumerate(new_spec)
   t2iAA[bb] = iAA
end

## 

