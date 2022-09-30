
import Base: ==, length

using Combinatorics: combinations, partitions

const BinDagNode = Tuple{Int, Int}

# struct BinDagNode
#    p1::Int 
#    p2::Int 
# end

struct AAEvalGraph
   nodes::Vector{BinDagNode}
   num1::Int
   numstore::Int
   projection::Vector{Int}
end

# warning: this is not the length of the basis!!! 
length(dag::AAEvalGraph) = length(dag.nodes)

# ==(dag1::AAEvalGraph, dag2::AAEvalGraph) = ACE1._allfieldsequal(dag1, dag2)

AAEvalGraph() = AAEvalGraph(Vector{BinDagNode}(undef, 0), 0, 0)

# # -------------- FIO

# # TODO: maybe there is a cleverer way to do this, for now this is just
# #       a quick hack to make sure it can be read without any ambiguity

# write_dict(gr::AAEvalGraph{T, TI}) where {T <: Real, TI <: Integer} =
#    Dict( "__id__" => "ACE1_AAEvalGraph",
#          "T" => write_dict(T),
#          "TI" => write_dict(TI),
#          "nodes1" => [ n[1] for n in gr.nodes ],
#          "nodes2" => [ n[2] for n in gr.nodes ],
#          "num1" => gr.num1,
#          "numstore" => gr.numstore
#       )


# function read_dict(::Val{:ACE1_AAEvalGraph}, D::Dict)
#    T = read_dict(D["T"])
#    TI = read_dict(D["TI"])
#    @assert T <: Real
#    @assert TI <: Integer
#    return AAEvalGraph{T, TI}(
#       collect(zip(D["nodes1"], D["nodes2"])),
#       D["num1"],
#       D["numstore"]
#    )
# end


# ---------------------------------------------------------------------
#   partition generator

_score_partition(p) = isempty(p) ? Inf : (1e9 * length(p) + maximum(p))

function _get_ns(p, specnew, specnew_dict)
   out = Vector{Int}(undef, length(p))
   for (i, kk_) in enumerate(p)
      if haskey(specnew_dict, kk_)
         out[i] = specnew_dict[kk_]
      else
         return Int[]
      end
   end
   return out
end


function _find_partition(kk, specnew, specnew_dict)
   worstp = _get_ns([ [k] for k in kk ], specnew, specnew_dict)
   @assert worstp == kk
   bestp = worstp
   bestscore = _score_partition(bestp)

   for ip in partitions(1:length(kk))
      p = _get_ns([ kk[i] for i in ip ], specnew, specnew_dict)
      score = _score_partition(p)
      if !isempty(p) && score < bestscore
         bestp = p
         bestscore = score
      end
   end

   return bestp
end


# return value is the number of fake nodes added to the dag
function _insert_partition!(nodes, specnew, specnew_dict,
                            kk, p,
                            ikk, specN)
   if length(p) == 2
      newnode = BinDagNode((p[1], p[2]))
      push!(nodes, newnode)
      push!(specnew, kk)
      specnew_dict[kk] = length(specnew)
      return 0
   else
      # @show kk, p
      # @infiltrate
      # reduce the partition by pushing a new node
      push!(nodes, BinDagNode((p[1], p[2])))
      kk1 = sort(vcat(specnew[p[1]], specnew[p[2]]))
      push!(specnew, kk1)
      specnew_dict[kk1] = length(specnew)
      # and now recurse with the reduced partition
      return 1 + _insert_partition!(nodes, specnew, specnew_dict,
                         kk, vcat( [length(nodes)], p[3:end] ),
                         ikk, specN)
   end
end

"""
Construct the DAG used to evaluate an AA basis and returns it as a `AAEvalGraph`

Arguments
* `spec` : AA basis specification, list of vectors of integers / indices pointing into A 

Kwargs: 
* `filter = _-> true` : 
* `verbose = false` : print some information about the 
"""
function AAEvalGraph(spec::AbstractVector; 
                     filter = _->true, verbose = false)
   @assert issorted(length.(spec))
   @assert all(issorted, spec)
   # we need to separate them into 1-p and many-p
   spec1 = spec[ length.(spec) .== 1 ]
   IN = (length(spec1)+1):length(spec)
   specN = spec[IN]

   # start assembling the dag
   nodes = BinDagNode[]
   sizehint!(nodes, length(spec))
   specnew = Vector{Int}[]
   specnew_dict = Dict{Vector{Int}, Int}()
   sizehint!(specnew, length(spec))

   # add the full 1-particle basis (N=1) into the dag
   num1 = maximum( maximum(vv) for vv in spec )
   for i = 1:num1
      push!(nodes, BinDagNode((i, 0)))
      push!(specnew, [i])
      specnew_dict[ [i] ] = length(specnew)
   end

   # now we can construct the rest
   extranodes = 0
   for (ikk, kk) in enumerate(specN)
      # find a good partition of kk
      p = _find_partition(kk, specnew, specnew_dict)
      extranodes += _insert_partition!(nodes, specnew, specnew_dict,
                                       kk, p, ikk, specN)
   end

   verbose && @info("Extra nodes inserted into the dag: $extranodes")
   numstore = length(nodes)
   num1old = num1

   projection = [ specnew_dict[vv] for vv in spec ]

   # re-organise the dag layout to minimise numstore
   # nodesfinal, num1, numstore = _reorder_dag!(nodes)

   return AAEvalGraph(nodes, num1, numstore, projection)
end


# TODO: this is currently not used; first need to add the functionality to 
#       to update specnew_dict i.e. the inverse mapping of specnew
function _reorder_dag!(nodes::Vector{BinDagNode})
   # collect all AA indices that are used anywhere in the dag
   newinds = zeros(Int, length(nodes))
   newnodes = BinDagNode[]

   # inds2 = stage-2 indices, i.e. temporary storage
   # inds3 = stage-3 indices, i.e. no intermediate storage
   inds2 = sort(unique([[ n[1] for n in nodes ]; [n[2] for n in nodes]]))

   # first add all 1p nodes
   for i = 1:length(nodes)
      n = nodes[i]
      if (n[2] == 0) ## && ((c != 0) || (i in inds2))
         @assert n[1] == i
         newinds[i] = i
         push!(newnodes, n)
      end
   end
   num1 = length(newnodes)

   # next add the remaining dependent nodes
   for i = 1:length(nodes)
      n = nodes[i]
      # not 1p basis && dependent node
      if (n[2] != 0) && (i in inds2)
         push!(newnodes, BinDagNode((newinds[n[1]], newinds[n[2]])))
         newinds[i] = length(newnodes)
      end
   end
   numstore = length(newnodes)

   # now go through one more time and add the independent nodes
   for i = 1:length(nodes)
      n = nodes[i]
      if (n[2] != 0) && (newinds[i] == 0)
         push!(newnodes, BinDagNode((newinds[n[1]], newinds[n[2]])))
         newinds[i] = length(newnodes)
      end
   end

   return newnodes, num1, numstore
end

