
using LoopVectorization



struct SparseSymmetricProduct{T}
   dag::AAEvalGraph
   pool_AA::ArrayCache{T, 1}
   ppool_AA::ArrayCache{T, 2}
   tmp_dag::TempArray{T, 1}
   ptmp_dag::TempArray{T, 2}
end


function SparseSymmetricProduct(spec::AbstractVector{<: AbstractVector}; T = Float64, kwargs...)
   dag = AAEvalGraph(spec; kwargs...)
   return SparseSymmetricProduct(
               dag, 
               ArrayCache{T, 1}(), ArrayCache{T, 2}(), 
               TempArray{T, 1}(), TempArray{T, 2}() )
end

# ------------------------- serial evaluation 


(basis::SparseSymmetricProduct)(A::AbstractArray) = evaluate(basis, A)

function evaluate(basis::SparseSymmetricProduct, A::AbstractVector{T}) where {T} 
   AAdag_ = acquire!(basis.tmp_dag, length(basis.dag), T)
   AAdag = parent(AAdag_)
   AA_ = acquire!(basis.pool_AA, length(basis.dag.projection), T)
   AA = parent(AA_)
   evaluate_dag!(AAdag, basis.dag, A)
   @inbounds for i = 1:length(basis.dag.projection)
      AA[i] = AAdag[basis.dag.projection[i]]
   end
   return AA_ 
end


function evaluate_dag!(AAdag, dag::AAEvalGraph, A)
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

# ------------------------- batched evaluation 


function evaluate(basis::SparseSymmetricProduct, A::AbstractMatrix{T}) where {T} 
   nX = size(A, 1)
   AAdag_ = acquire!(basis.ptmp_dag, (nX, length(basis.dag)), T)
   AAdag = parent(AAdag_)
   AA_ = acquire!(basis.ppool_AA, (nX, length(basis.dag.projection)), T)
   AA = parent(AA_)
   evaluate_dag!(AAdag, basis.dag, A)
   @inbounds for j = 1:length(basis.dag.projection)
      jj = basis.dag.projection[j]
      for i = 1:nX
         AA[i, j] = AAdag[i, jj]
      end
   end
   return AA_
end

function evaluate_dag!(AAdag, dag::AAEvalGraph, A::AbstractMatrix{T}) where {T} 
   nX = size(A, 1)
   nodes = dag.nodes
   @assert size(AAdag, 2) >= length(dag)
   @assert size(AAdag, 1) >= size(A, 1)
   @assert size(A, 2) >= dag.num1

   # Stage-1: copy the 1-particle basis into AAdag
   @inbounds begin 
      for i = 1:dag.num1
         # if (T <: Real)
         @simd ivdep for j = 1:nX
            AAdag[j, i] = A[j, i]
         end
         # else 
         #    for j = 1:nX
         #       AAdag[j, i] = A[j, i]
         #    end
         # end
      end

   # Stage-2: go through the dag and store the intermediate results we need
      for i = (dag.num1+1):length(dag)
         n1, n2 = nodes[i]
         # if (T <: Real)
         @simd ivdep for j = 1:nX 
            AAdag[j, i] = AAdag[j, n1] * AAdag[j, n2]
         end
         # else
         #    for j = 1:nX 
         #       AAdag[j, i] = AAdag[j, n1] * AAdag[j, n2]
         #    end
         # end
      end
   end

   return AAdag
end




# ------------------------- linear model evaluation 


function contract(w, basis::SparseSymmetricProduct, 
                  A::AbstractArray{T}) where {T}
   AA = evaluate(basis, A)
   out = w * AA 
   release!(AA) 
   return out 
end

function contract_ed(w, 
                     basis::SparseSymmetricProduct, 
                     A::AbstractVector{<: Number})
   Δ = zeros(eltype(w), length(basis.dag))
   @inbounds for i = 1:length(basis.dag.projection)
      Δ[basis.dag.projection[i]] = w[i]
   end
   # Δ[basis.dag.projection] .= w[:]
   val, pb = _rruleA_contract_dag(Δ', basis, A)
   return val, pb(Δ)
end

# at the moment this only produces the pullback w.r.t. A 
# but not w.r.t. the basis! 
function _rruleA_contract_dag(w, basis::SparseSymmetricProduct, 
                              A::AbstractVector{T}) where {T}
   # forward pass    
   AAdag_ = acquire!(basis.tmp_dag, length(basis.dag))
   AAdag = parent(AAdag_)
   evaluate_dag!(AAdag, basis.dag, A)
   val = w * AAdag

   T∂ = promote_type(T, eltype(w), eltype(AAdag))
   ∂A = zeros(T∂, length(A))

   # reverse pass is implemented in pullback_contract_dag
   return val, Δ -> pullback_evaluate_dag!(∂A, Δ, basis, AAdag)
end

# this is the simplest case for the pull-back, when the cotangent is just a 
# scalar and there is only a single input. 
# note that in executing this, we are chaning Δ! This means that the 
# caller has to make sure it will not be used afterwards. 
function pullback_evaluate_dag!(∂A, 
                  ∂AAdag::AbstractVector, 
                  basis::SparseSymmetricProduct, AA::AbstractVector)
   dag = basis.dag                   
   nodes = dag.nodes
   num1 = dag.num1 
   @assert length(AA) >= length(dag)
   @assert length(nodes) >= length(dag)
   @assert length(∂AAdag) >= length(dag)
   @assert length(∂A) >= num1

   TΔ = promote_type(eltype(∂AAdag), eltype(AA))
   Δ̃ = zeros(TΔ, length(dag))
   @inbounds for i = 1:length(dag)
      Δ̃[i] = ∂AAdag[i]
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

   return ∂A                                                         
end


# ------------------------- linear model evaluation - DAG based 


function pullback_evaluate_dag!(
                     ∂A::AbstractMatrix, 
                     ∂AA::AbstractMatrix, 
                     basis::SparseSymmetricProduct, 
                     AA::AbstractMatrix, 
                     nX = size(AA, 1))
   dag = basis.dag                     
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

   end

   return nothing 
end

