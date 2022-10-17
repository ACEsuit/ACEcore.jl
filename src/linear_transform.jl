


struct LinearTransform{TT}
   T::TT
end


function evaluate!(BB, lt::LinearTransform, AA::AbstractVector) 
   mul!(BB, lt.T, AA)
   return nothing 
end


function evaluate!(BB, lt::LinearTransform, AA::AbstractMatrix) 
   mul!(transpose(BB), lt.T, transpose(AA))
   return nothing 
end
