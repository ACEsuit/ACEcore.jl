

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


evaluate(basis::PooledSparseProduct, BB...) = 
         evaluate!(nothing, basis, BB...)

test_evaluate(basis::PooledSparseProduct, BB::Tuple) = 
       [ prod(BB[j][basis.spec[i][j]] for j = 1:length(BB)) 
            for i = 1:length(basis) ]

# function evaluate(basis::PooledSparseProduct, cfg::UConfig)
#    @assert length(cfg) > 0 "PooledSparseProduct can only be evaluated with non-empty configurations"
#    # evaluate the first item "manually", then so we know the output types 
#    # but then write directly into the allocated array to avoid additional 
#    # allocations. 
#    A = evaluate(basis, first(cfg))
#    for (i, X) in enumerate(cfg)
#       i == 1 && continue; 
#       add_into_A!(A, basis, X)
#    end
#    return A 
# end 

# function evaluate!(A::AbstractVector, basis::PooledSparseProduct, X::AbstractState)
#    fill!(A, zero(eltype(A)))
#    add_into_A!(A, basis, X)
#    return A
# end

# function evaluate!(A, basis::PooledSparseProduct, cfg::UConfig)
#    fill!(A, zero(eltype(A)))
#    for X in cfg 
#       add_into_A!(A, basis, X)
#    end
#    return A
# end


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

@generated function prod_and_pool!(A::VA, basis::PooledSparseProduct{NB}, BB) where {NB, VA}
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
         @avx for j = 1:nX
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


