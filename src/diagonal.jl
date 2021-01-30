
###
# Lmul
####

mulreduce(M::Mul{<:DiagonalLayout,<:DiagonalLayout}) = Lmul(M)
mulreduce(M::Mul{<:DiagonalLayout}) = Lmul(M)
mulreduce(M::Mul{<:Any,<:DiagonalLayout}) = Rmul(M)

# Diagonal multiplication never changes structure
similar(M::Lmul{<:DiagonalLayout}, ::Type{T}, axes) where T = similar(M.B, T, axes)
# equivalent to rescaling
function materialize!(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout}})
    M.B .= getindex_value(M.A.diag) .* M.B
    M.B
end


copy(M::Lmul{<:DiagonalLayout,<:DiagonalLayout}) = Diagonal(diagonaldata(M.A) .* diagonaldata(M.B))
copy(M::Lmul{<:DiagonalLayout}) = diagonaldata(M.A) .* M.B
copy(M::Rmul{<:Any,<:DiagonalLayout}) = M.A .* permutedims(diagonaldata(M.B))




# Diagonal multiplication never changes structure
similar(M::Rmul{<:Any,<:DiagonalLayout}, ::Type{T}, axes) where T = similar(M.A, T, axes)
# equivalent to rescaling
function materialize!(M::Rmul{<:Any,<:DiagonalLayout{<:AbstractFillLayout}})
    M.A .= M.A .* getindex_value(M.B.diag)
    M.A
end


function materialize!(M::Ldiv{<:DiagonalLayout})
    M.B .= M.A.diag .\ M.B
    M.B
end

copy(M::Ldiv{<:DiagonalLayout,<:DiagonalLayout}) = Diagonal(inv.(M.A.diag) .* M.B.diag)
copy(M::Ldiv{<:DiagonalLayout}) = inv.(M.A.diag) .* M.B
copy(M::Ldiv{<:DiagonalLayout{<:AbstractFillLayout}}) = inv(getindex_value(M.A.diag)) .* M.B
copy(M::Ldiv{<:DiagonalLayout{<:AbstractFillLayout},<:DiagonalLayout}) = Diagonal(inv(getindex_value(M.A.diag)) .* M.B.diag)

copy(M::Rdiv{<:DiagonalLayout,<:DiagonalLayout}) = Diagonal(M.A.diag .* inv.(M.B.diag))
copy(M::Rdiv{<:Any,<:DiagonalLayout}) = M.A .* inv.(permutedims(M.B.diag))
copy(M::Rdiv{<:Any,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A .* inv(getindex_value(M.B.diag)) 
copy(M::Rdiv{<:DiagonalLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = Diagonal(M.A.diag .* inv(getindex_value(M.B.diag)))