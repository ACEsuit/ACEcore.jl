
using LoopVectorization

struct SparseSymmProd{T} 
   dag::SparseSymmProdDAG
   # replace projection with linear transform?
   # proj::LinearTransform{SparseMatrixCSC{Bool, Int64}}
   proj::Vector{Int}
   pool_AA::ArrayCache{T, 1}
   bpool_AA::ArrayCache{T, 2}
end

function SparseSymmProd(spec::AbstractVector{<: AbstractVector}; T = Float64, kwargs...)
   dag = SparseSymmProdDAG(spec; T=T, kwargs...)
   return SparseSymmProd(dag, dag.projection, 
                         ArrayCache{T, 1}(), ArrayCache{T, 2}())
end

Base.length(basis::SparseSymmProd) = length(basis.proj)

(basis::SparseSymmProd)(args...) = evaluate(basis, args...)


# -------------- evaluation interfaces 

function evaluate(basis::SparseSymmProd, A::AbstractVector{T}) where {T}
   AA = acquire!(basis.pool_AA, length(basis))
   evaluate!(parent(AA), basis, A)
   return AA
end

function evaluate(basis::SparseSymmProd, A::AbstractMatrix{T}) where {T}
   nX = size(A, 1)
   AA = acquire!(basis.bpool_AA, (nX, length(basis)))
   evaluate!(parent(AA), basis, A)
   return AA
end


# -------------- kernels  (these are really just interfaces as well...)

# this one does both batched and unbatched
function evaluate!(AA, basis::SparseSymmProd, A)
   AAdag = evaluate(basis.dag, A)
   _project!(AA, basis.proj, AAdag)
   release!(AAdag)
   return AA 
end

# serial projection 
function _project!(BB, proj::Vector{<: Integer}, AA::AbstractVector)
   @inbounds for i = 1:length(proj)
      BB[i] = AA[proj[i]]
   end
   return nothing
end

# batched projection 
function _project!(BB, proj::Vector{<: Integer}, AA::AbstractMatrix)
   nX = size(AA, 1)
   @assert size(BB, 1) >= nX
   @inbounds for i = 1:length(proj)
      p_i = proj[i]
      for j = 1:nX
         BB[j, i] = AA[j, p_i]
      end
   end
   return nothing
end


