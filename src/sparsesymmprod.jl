
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
   evaluate_AAdag!(AAdag, basis.dag, A)
   @inbounds for i = 1:length(basis.dag.projection)
      AA[i] = AAdag[basis.dag.projection[i]]
   end
   return AA_ 
end


function evaluate_AAdag!(AAdag, dag::AAEvalGraph, A)
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
   evaluate_AAdag!(AAdag, basis.dag, A)
   @inbounds for j = 1:length(basis.dag.projection)
      jj = basis.dag.projection[j]
      for i = 1:nX
         AA[i, j] = AAdag[i, jj]
      end
   end
   return AA_
end

function evaluate_AAdag!(AAdag, dag::AAEvalGraph, A::AbstractMatrix{T}) where {T} 
   nX = size(A, 1)
   nodes = dag.nodes
   @assert size(AAdag, 2) >= length(dag)
   @assert size(AAdag, 1) >= size(A, 1)
   @assert size(A, 2) >= dag.num1

   # Stage-1: copy the 1-particle basis into AAdag
   @inbounds begin 
      for i = 1:dag.num1
         if (T <: Real)
            @avx for j = 1:nX
               AAdag[j, i] = A[j, i]
            end
         else 
            for j = 1:nX
               AAdag[j, i] = A[j, i]
            end
         end
      end

   # Stage-2: go through the dag and store the intermediate results we need
      for i = (dag.num1+1):length(dag)
         n1, n2 = nodes[i]
         if (T <: Real)
            @avx for j = 1:nX 
               AAdag[j, i] = AAdag[j, n1] * AAdag[j, n2]
            end
         else
            for j = 1:nX 
               AAdag[j, i] = AAdag[j, n1] * AAdag[j, n2]
            end
         end
      end
   end

   return AAdag
end


