module ACEcore

using ObjectPools: ArrayCache, TempArray, acquire!, release!

include("sparseprod.jl")

include("symmprod_dag.jl")
include("symmprod_dag_kernels.jl")

include("simpleprodbasis.jl")
include("sparsesymmprod.jl")

include("testing.jl")

end
