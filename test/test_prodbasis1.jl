
using ACEcore, BenchmarkTools
using ACEcore: SimpleProdBasis
using ACEcore: ProdBasis

##

M = 20 
spec = ACEcore.Testing.generate_SO2_spec(5, M)
A = randn(ComplexF64, 2*M+1)

basis1 = SimpleProdBasis(spec)
AA1 = basis1(A)

##

basis2 = ProdBasis(spec)
AA2 = basis2(A)

@btime $basis1($A)
@btime $basis2($A)

AA1 ≈ AA2