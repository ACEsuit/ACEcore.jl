
using ACEcore, BenchmarkTools
using ACEcore: SimpleProdBasis, ProdBasis, ProdBasis2

##

M = 20 
spec = ACEcore.Testing.generate_SO2_spec(5, M)
A = randn(ComplexF64, 2*M+1)

basis1 = SimpleProdBasis(spec)
AA1 = basis1(A)

##

basis2 = ProdBasis(spec)
AA2 = basis2(A)

basis3 = ProdBasis2(spec; T = ComplexF64)
AA3 = basis3(A)

@btime $basis1($A)
@btime $basis2($A)
@btime $basis3($A)
@btime (AA = $basis3($A); ACEcore.release!(AA))

AA1 ≈ AA2 ≈ AA3 


##

@profview let basis = basis3, A = A 
   for n = 1:1000_000 
      AA = basis(A)
      ACEcore.release!(AA)
   end
end

