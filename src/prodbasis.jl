

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
   AA = zeros(eltype(A), length(basis.dag))
   evaluate!(AA, basis.dag, A)
   return AA
end


function evaluate!(AAdag, dag::AAEvalGraph, A)
   nodes = dag.nodes
   @assert length(AAdag) >= dag.numstore
   @assert length(A) >= dag.num1

   # Stage-1: copy the 1-particle basis into AAdag
   # @inbounds 
   for i = 1:dag.num1
      AAdag[i] = A[i]
   end

   # Stage-2: go through the dag and store the intermediate results we need
   # @inbounds 
   for i = (dag.num1+1):length(dag)
      n1, n2 = nodes[i]
      AAdag[i] = AAdag[n1] * AAdag[n2]
   end

   return AAdag 
end
