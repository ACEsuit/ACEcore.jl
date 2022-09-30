module ACEcore

using ObjectPools: ArrayCache, TempArray, acquire!, release!

include("simpleprodbasis.jl")

include("prodbasis_dag.jl")
include("spsymprod.jl")

include("testing.jl")

end
