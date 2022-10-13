

struct PooledSparseProduct{NB}
   spec::Vector{NTuple{NB, Int}}
   # ---- temporaries 
end

function PooledSparseProduct()
   return PooledSparseProduct(bases, NTuple{NB, Int}[])
end

# each column defines a basis element
function PooledSparseProduct(spec::Matrix{<: Integer})
   @assert all(spec .> 0)
   spect = [ Tuple(spec[:, i]...) for i = 1:size(spec, 2) ]
   return PooledSparseProduct(spect)
end

Base.length(basis::PooledSparseProduct) = length(basis.spec)

# function Base.show(io::IO, basis::PooledSparseProduct)
#    print(io, "PooledSparseProduct(")
#    print(io, basis.bases)
# end


# ----------------------- evaluation interfaces 


evaluate(basis::PooledSparseProduct, BB::Tuple) = 
         evaluate!(nothing, basis, BB::Tuple)

evalpool(basis::PooledSparseProduct, BB::Tuple) = 
         evalpool!(nothing, basis, BB::Tuple)

test_evaluate(basis::PooledSparseProduct, BB::Tuple) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]


# ----------------------- evaluation kernels 

import Base.Cartesian: @nexprs

function _write_A_code(VA, NB)
   prodBi_str = "BB[1][ϕ[1]]" 
   for i in 2:NB
      prodBi_str *= " * BB[$i][ϕ[$i]]"
   end
   prodBi = Meta.parse(prodBi_str)
   if VA == Nothing 
      getVT = "promote_type(" * prod("eltype(BB[$i]), " for i = 1:NB) * ")"
      getA = Meta.parse("_A = zeros($(getVT), length(basis))")
   else 
      getA = :(_A = A)
   end
   return prodBi, getA 
end

@generated function evaluate!(A::VA, basis::PooledSparseProduct{NB}, BB) where {NB, VA}
   prodBi, getA = _write_A_code(VA, NB)
   quote
      @assert length(BB) == $NB
      # allocate A if necessary or just name _A = A if A is a buffer 
      $(getA)
      # evaluate the 1p product basis functions and add/write into _A
      for (iA, ϕ) in enumerate(basis.spec)
         @inbounds _A[iA] += $prodBi 
      end
      return _A
   end
end

function _write_A_code_pool(VA, NB)
   prodBi_str = "BB[1][j, ϕ1]" 
   for i in 2:NB
      prodBi_str *= " * BB[$i][j, ϕ$i]"
   end
   prodBi = Meta.parse(prodBi_str)
   if VA == Nothing 
      getVT = "promote_type(" * prod("eltype(BB[$i]), " for i = 1:NB) * ")"
      getA = Meta.parse("_A = zeros($(getVT), length(basis))")
   else 
      getA = :(_A = A)
   end
   return prodBi, getA 
end

@generated function evalpool!(A::VA, basis::PooledSparseProduct{NB}, BB) where {NB, VA}
   prodBi, getA = _write_A_code_pool(VA, NB)
   quote
      # @assert length(BB) == $NB
      nX = size(BB[1], 1)
      # @assert all( size(B, 1) == nX for B in BB )

      # allocate A if necessary or just name _A = A if A is a buffer 
      $(getA)
      a = zero(eltype(_A))

      # evaluate the 1p product basis functions and add/write into _A
      @inbounds for (iA, ϕ) in enumerate(basis.spec)
         a *= 0 
         ϕ1 = ϕ[1]; ϕ2 = ϕ[2]; ϕ3 = ϕ[3]
         @simd ivdep for j = 1:nX  # @avx or @simd ivdep 
            a += $(prodBi)
         end
         _A[iA] = a
      end
      return _A
   end
end

# this code should never be used, we keep it just for testing 
# the performance of the generated code. 
# function prod_and_pool3!(A::VA, basis::PooledSparseProduct{3}, 
#                      BB::Tuple{TB1, TB2, TB3}) where {VA, TB1, TB2, TB3}
#    B1 = BB[1]; B2 = BB[2]; B3 = BB[3] 
#    # VT = promote_type(eltype(B1), eltype(B2), eltype(B3))
#    nX = size(B1, 1) 
#    @assert size(B2, 1) == size(B3, 1) == nX 
#    a = zero(eltype(A))

#    @inbounds for (iA, ϕ) in enumerate(basis.spec)
#       a *= 0
#       ϕ1 = ϕ[1]; ϕ2 = ϕ[2]; ϕ3 = ϕ[3]
#       @avx for j = 1:nX 
#          a += B1[j, ϕ1] * B2[j, ϕ2] * B3[j, ϕ3]
#       end
#       A[iA] = a
#    end
#    return A
# end


# -------------------- reverse mode gradient

using StaticArrays

@generated function _prod_grad(b::SVector{1, T}) where {T} 
   quote
      return b[1], SVector(one(T))
   end
end

function _code_prod_grad(NB, T)
   code = Expr[] 
   push!(code, :(g2 = b[1]))
   for i = 3:NB 
      push!(code, Meta.parse("g$i = g$(i-1) * b[$(i-1)]"))
   end
   push!(code, Meta.parse("val = g$NB * b[$NB]"))
   push!(code, Meta.parse("h = b[$NB]"))
   for i = NB-1:-1:2
      push!(code, Meta.parse("g$i *= h"))
      push!(code, Meta.parse("h *= b[$i]"))
   end
   push!(code, :(g1 = h))
   push!(code, Meta.parse(
            "g = SVector(" * join([ "g$i" for i = 1:NB ], ", ") * ")" ))
   push!(code, :( return val, g))
end

@generated function _prod_grad(b::SVector{NB, T}) where {NB, T} 
   code = _code_prod_grad(NB, T)
   quote
      $(code...)
   end
end

using Base.Cartesian: @nexprs


function _rrule_evalpool(basis::PooledSparseProduct{NB}, BB::Tuple) where {NB}
   A = evalpool(basis, BB)
   return A, ∂A -> _pullback_evalpool(∂A, basis, BB)
end


function _pullback_evalpool(∂A, basis::PooledSparseProduct{NB}, BB::Tuple) where {NB}

   nX = size(BB[1], 1)
   @assert all(nX == size(BB[i], 1) for i = 1:NB)
   @assert length(∂A) == length(basis)
   @assert length(BB) == NB 
   
   TA = promote_type(eltype.(BB)...)
   ∂BB = ntuple(i -> zeros(TA, size(BB[i])...), NB)

   for (iA, ϕ) in enumerate(basis.spec)
      ∂A_iA = ∂A[iA]
      for j = 1:nX 
         b = SVector( ntuple(i -> BB[i][j, ϕ[i]], NB) )
         _, g = _prod_grad(b)

         # write into ∂BB
         # @nexprs NB i -> ∂BB[i][j, ϕ[i]] += ∂A_iA * g[i]
         for i = 1:NB
            ∂BB[i][j, ϕ[i]] += ∂A_iA * g[i]
         end
      end 
   end

   return ∂BB
end
