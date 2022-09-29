using LoopVectorization

# TODO: rename ProdBasis -> Sparse Symmetric Tensor Product 



"""
Recursive implementation of the ACE product basis, this should be the main 
code to be used on CPU architectures. 
"""
struct ProdBasis
   dag::AAEvalGraph
end


# function get_spec(basis::ProdBasis, i)
#    return basis.spec
# end
function ProdBasis(spec::AbstractVector{<: AbstractVector}; kwargs...)
   dag = AAEvalGraph(spec; kwargs...)
   return ProdBasis(dag)
end

# ---------------------------------------------------------------------
#   evaluation codes

(basis::ProdBasis)(A::AbstractVector) = evaluate(basis, A)

function evaluate(basis::ProdBasis, A::AbstractVector)
   AAdag = zeros(eltype(A), length(basis.dag))
   evaluate!(AAdag, basis.dag, A)
   return AAdag[basis.dag.projection]
end


function evaluate!(AAdag, dag::AAEvalGraph, A)
   nodes = dag.nodes
   @assert length(AAdag) >= dag.numstore
   @assert length(A) >= dag.num1

   # Stage-1: copy the 1-particle basis into AAdag
   @inbounds for i = 1:dag.num1
      AAdag[i] = A[i]
   end

   # Stage-2: go through the dag and store the intermediate results we need
   @inbounds for i = (dag.num1+1):length(dag)
      n1, n2 = nodes[i]
      AAdag[i] = AAdag[n1] * AAdag[n2]
   end

   return AAdag
end


# ---------------------------------------------------------------------

import Polynomials4ML: TempArray, ArrayCache, acquire!, release!

struct ProdBasis2{T}
   dag::AAEvalGraph
   pool_AA::ArrayCache{T, 1}
   tmp_dag::TempArray{T, 1}
end


function ProdBasis2(spec::AbstractVector{<: AbstractVector}; T = Float64, kwargs...)
   dag = AAEvalGraph(spec; kwargs...)
   return ProdBasis2(dag, ArrayCache{T, 1}(), TempArray{T, 1}() )
end


(basis::ProdBasis2)(A::AbstractVector) = evaluate(basis, A)

function evaluate(basis::ProdBasis2, A::AbstractVector{T}) where {T} 
   AAdag_ = acquire!(basis.tmp_dag, length(basis.dag), T)
   AAdag = parent(AAdag_)
   AA_ = acquire!(basis.pool_AA, length(basis.dag.projection), T)
   AA = parent(AA_)
   evaluate!(AAdag, basis.dag, A)
   @inbounds for i = 1:length(basis.dag.projection)
      AA[i] = AAdag[basis.dag.projection[i]]
   end
   return AA_ 
   # return AAdag[basis.dag.projection]
end

