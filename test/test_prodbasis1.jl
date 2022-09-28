
using ACEcore
using ACEcore: SimpleProdBasis
using ACEcore: ProdBasis

##

M = 10 
spec = ACEcore.Testing.generate_SO2_spec(5, 10)
A = randn(ComplexF64, 2*M+1)

basis1 = SimpleProdBasis(spec)
basis1(A)

##

basis2 = ProdBasis(spec)
basis2(A)

