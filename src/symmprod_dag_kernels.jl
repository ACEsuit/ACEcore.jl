

# --------------- interface functions 

(dag::SparseSymmProdDAG)(args...) = evaluate(dag, args...)

function evaluate(dag::SparseSymmProdDAG, A::AbstractVector{T}) where {T}
   AA = acquire!(dag.pool_AA, length(dag), T)
   evaluate!(AA, dag, A)
   return AA
end

function evaluate(dag::SparseSymmProdDAG, A::AbstractMatrix{T}) where {T}
   nX = size(A, 1)
   AA = acquire!(dag.bpool_AA, (nX, length(dag)), T)
   evaluate!(AA, dag, A)
   return AA
end


# --------------- old evaluation 

function evaluate!(AA, dag::SparseSymmProdDAG, A)
   nodes = dag.nodes
   @assert length(AA) >= dag.numstore
   @assert length(A) >= dag.num1

   # Stage-1: copy the 1-particle basis into AA
   @inbounds for i = 1:dag.num1
      AA[i] = A[i]
   end

   # Stage-2: go through the dag and store the intermediate results we need
   @inbounds for i = (dag.num1+1):length(dag)
      n1, n2 = nodes[i]
      AA[i] = AA[n1] * AA[n2]
   end

   return nothing
end


# this is the simplest case for the pull-back, when the cotangent is just a 
# scalar and there is only a single input. 
# note that in executing this, we are changing ∂AAdag. This means that the 
# caller has to make sure it will not be used afterwards. 
#
# Warning (to be documented!!!) : the input must be AA and not A!!!
#                    A is no longer needed to evaluate the pullback
#
function pullback_arg!(∂A, ∂AA::AbstractVector, 
                       dag::SparseSymmProdDAG, AA::AbstractVector)
   nodes = dag.nodes
   num1 = dag.num1 
   @assert length(AA) >= length(dag)
   @assert length(nodes) >= length(dag)
   @assert length(∂AA) >= length(dag)
   @assert length(∂A) >= num1

   TΔ = promote_type(eltype(∂AA), eltype(AA))
   Δ̃ = zeros(TΔ, length(dag))
   @inbounds for i = 1:length(dag)
   Δ̃[i] = ∂AA[i]
   end

   # BACKWARD PASS
   # --------------
   for i = length(dag):-1:num1+1
      wi = Δ̃[i]
      n1, n2 = nodes[i]
      Δ̃[n1] = muladd(wi, AA[n2], Δ̃[n1])
      Δ̃[n2] = muladd(wi, AA[n1], Δ̃[n2])
   end

   # at this point the Δ̃[i] for i = 1:num1 will contain the 
   # gradients w.r.t. A 
   for i = 1:num1 
      ∂A[i] = Δ̃[i]
   end

   return nothing                                                    
end


# ------------------------- batched kernels 


function evaluate!(AA, dag::SparseSymmProdDAG, A::AbstractMatrix{T}) where {T} 
   nX = size(A, 1)
   nodes = dag.nodes
   @assert size(AA, 2) >= length(dag)
   @assert size(AA, 1) >= size(A, 1)
   @assert size(A, 2) >= dag.num1

   # Stage-1: copy the 1-particle basis into AA
   @inbounds begin 
      for i = 1:dag.num1
         # if (T <: Real)
         @simd ivdep for j = 1:nX
            AA[j, i] = A[j, i]
         end
      end

   # Stage-2: go through the dag and store the intermediate results we need
      for i = (dag.num1+1):length(dag)
         n1, n2 = nodes[i]
         # if (T <: Real)
         @simd ivdep for j = 1:nX 
            AA[j, i] = AA[j, n1] * AA[j, n2]
         end
      end
   end # inbounds 

   return nothing 
end



function pullback_arg!(∂A::AbstractMatrix, 
                       ∂AA::AbstractMatrix, 
                       dag::SparseSymmProdDAG,
                       AA::AbstractMatrix, 
                       nX = size(AA, 1))
   nodes = dag.nodes
   num1 = dag.num1 
   @assert size(AA, 2) >= length(dag)
   @assert size(∂AA, 2) >= length(dag)
   @assert size(∂A, 2) >= num1
   @assert size(∂A, 1) >= nX 
   @assert size(∂AA, 1) >= nX 
   @assert size(AA, 1) >= nX 
   @assert length(nodes) >= length(dag)

   @inbounds begin 

      for i = length(dag):-1:num1+1
         n1, n2 = nodes[i]
         @simd ivdep for j = 1:nX 
            wi = ∂AA[j, i]
            ∂AA[j, n1] = muladd(wi, AA[j, n2], ∂AA[j, n1])
            ∂AA[j, n2] = muladd(wi, AA[j, n1], ∂AA[j, n2])
         end
      end

      # at this point the Δ̃[i] for i = 1:num1 will contain the 
      # gradients w.r.t. A 
      for i = 1:num1 
         @simd ivdep for j = 1:nX 
            ∂A[j, i] = ∂AA[j, i]
         end
      end

   end # inbounds 

   return nothing 
end

