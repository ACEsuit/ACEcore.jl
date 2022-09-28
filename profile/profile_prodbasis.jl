using BenchmarkTools, LoopVectorization 

struct ProdBasis 
   lenA::Int   
   p1::Vector{Int}
   p2::Vector{Int}
end

Base.length(basis::ProdBasis) = basis.lenA + length(basis.p1) 

_alloc(basis::ProdBasis) = zeros(Float64, length(basis))

function rand_ProdBasis(lenA, lenAA)
   p1 = zeros(Int, lenAA - lenA)
   p2 = zeros(Int, lenAA - lenA)
   for i = 1:length(p1)
      p1[i] = rand(1:max(1, (i ÷ 2)))
      p2[i] = rand(1:max(1, (i ÷ 2)))
   end
   return ProdBasis(lenA, p1, p2)
end

function evaluate!(AA, basis::ProdBasis, A::AbstractVector) 
   @assert length(AA) >= length(basis)
   lenA = length(A) 
   @inbounds begin 
      for i = 1:lenA 
         AA[i] = A[i] 
      end 
      for i = 1:length(basis.p1) 
         AA[lenA+i] = AA[basis.p1[i]] * AA[basis.p2[i]]
      end
   end 
   return AA 
end 

function evaluate!(AA, basis::ProdBasis, A::AbstractMatrix) 
   nX = size(A, 1)
   @assert size(AA, 2) == length(basis)
   @assert size(AA, 1) == nX
   lenA = size(A, 2)
   @inbounds begin 
      for n = 1:lenA 
         for i = 1:nX 
            AA[i, n] = A[i, n] 
         end
      end 
      for n = 1:length(basis.p1) 
         i_AA0 = nX * (n-1 + lenA)
         i_p10 = nX * (basis.p1[n] - 1)
         i_p20 = nX * (basis.p2[n] - 1)
         for i = 1:nX 
            AA[i_AA0 + i] = AA[i_p10 + i] * AA[i_p20 + i]
         end
      end
   end
   return AA 
end 

##

lenA = 500 
lenAA = 10_000 

basis = rand_ProdBasis(lenA, lenAA)
A = randn(lenA)
AA = _alloc(basis)
cA = randn(ComplexF64, lenA)
cAA = zeros(ComplexF64, lenAA)

evaluate!(AA, basis, A)


@btime evaluate!($AA, $basis, $A)
@btime evaluate!($cAA, $basis, $cA)

## 

nX = 1000
Ab = randn(nX, lenA)
AAb = zeros(nX, lenAA)
cAb = randn(ComplexF64, nX, lenA)
cAAb = zeros(ComplexF64, nX, lenAA)

evaluate!(AAb, basis, Ab)


@btime evaluate!($AAb, $basis, $Ab)
@btime evaluate!($cAAb, $basis, $cAb)

##

using LinearAlgebra
ccb = randn(nX, lenAA)

@btime dot($ccb, $AAb)