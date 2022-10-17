function contract(w, basis::SparseSymmetricProduct, 
   A::AbstractArray{T}) where {T}
AA = evaluate(basis, A)
out = w * AA 
release!(AA) 
return out 
end

function contract_ed(w, 
      basis::SparseSymmetricProduct, 
      A::AbstractVector{<: Number})
Δ = zeros(eltype(w), length(basis.dag))
@inbounds for i = 1:length(basis.dag.projection)
Δ[basis.dag.projection[i]] = w[i]
end
# Δ[basis.dag.projection] .= w[:]
val, pb = _rruleA_contract_dag(Δ', basis, A)
return val, pb(Δ)
end