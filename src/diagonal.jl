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


copy(M::Lmul{<:DiagonalLayout,<:DiagonalLayout}) = diagonal(diagonaldata(M.A) .* diagonaldata(M.B))
copy(M::Lmul{<:DiagonalLayout}) = diagonaldata(M.A) .* M.B
copy(M::Rmul{<:Any,<:DiagonalLayout}) = M.A .* permutedims(diagonaldata(M.B))


dualadjoint(_) = adjoint
dualadjoint(::Transpose) = transpose
dualadjoint(V::SubArray) = dualadjoint(parent(V))
function copy(M::Rmul{<:DualLayout,<:DiagonalLayout})
    adj = dualadjoint(M.A)
    adj(adj(M.B) * adj(M.A))
end



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

copy(M::Ldiv{<:DiagonalLayout,<:DiagonalLayout}) = diagonal(M.A.diag .\ M.B.diag)
copy(M::Ldiv{<:DiagonalLayout}) = M.A.diag .\ M.B
copy(M::Ldiv{<:DiagonalLayout{<:AbstractFillLayout}}) = inv(getindex_value(M.A.diag)) .* M.B
copy(M::Ldiv{<:DiagonalLayout{<:AbstractFillLayout},<:DiagonalLayout}) = diagonal(inv(getindex_value(M.A.diag)) .* M.B.diag)

copy(M::Rdiv{<:DiagonalLayout,<:DiagonalLayout}) = diagonal(M.A.diag .* inv.(M.B.diag))
copy(M::Rdiv{<:Any,<:DiagonalLayout}) = M.A .* inv.(permutedims(M.B.diag))
copy(M::Rdiv{<:Any,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A .* inv(getindex_value(M.B.diag))
copy(M::Rdiv{<:DiagonalLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = diagonal(M.A.diag .* inv(getindex_value(M.B.diag)))



## bi/tridiagonal copy
copy(M::Rmul{<:BidiagonalLayout,<:DiagonalLayout}) = convert(Bidiagonal, M.A) * M.B
copy(M::Lmul{<:DiagonalLayout,<:BidiagonalLayout}) = M.A * convert(Bidiagonal, M.B)
copy(M::Rmul{<:TridiagonalLayout,<:DiagonalLayout}) = convert(Tridiagonal, M.A) * M.B
copy(M::Lmul{<:DiagonalLayout,<:TridiagonalLayout}) = M.A * convert(Tridiagonal, M.B)
copy(M::Rmul{<:SymTridiagonalLayout,<:DiagonalLayout}) = convert(SymTridiagonal, M.A) * M.B
copy(M::Lmul{<:DiagonalLayout,<:SymTridiagonalLayout}) = M.A * convert(SymTridiagonal, M.B)

copy(M::Lmul{DiagonalLayout{OnesLayout}}) = _copy_oftype(M.B, eltype(M))
copy(M::Lmul{DiagonalLayout{OnesLayout},<:DiagonalLayout}) = Diagonal(_copy_oftype(diagonaldata(M.B), eltype(M)))
copy(M::Lmul{<:DiagonalLayout,DiagonalLayout{OnesLayout}}) = Diagonal(_copy_oftype(diagonaldata(M.A), eltype(M)))
copy(M::Lmul{DiagonalLayout{OnesLayout},DiagonalLayout{OnesLayout}}) = _copy_oftype(M.B, eltype(M))
copy(M::Rmul{<:Any,DiagonalLayout{OnesLayout}}) = _copy_oftype(M.A, eltype(M))

copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout}}) = getindex_value(diagonaldata(M.A)) * M.B
copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout},<:DiagonalLayout}) = getindex_value(diagonaldata(M.A)) * M.B
copy(M::Rmul{<:Any,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A * getindex_value(diagonaldata(M.B))

copy(M::Rmul{<:BidiagonalLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A * getindex_value(diagonaldata(M.B))
copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout},<:BidiagonalLayout}) = getindex_value(diagonaldata(M.A)) * M.B
copy(M::Rmul{<:TridiagonalLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A * getindex_value(diagonaldata(M.B))
copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout},<:TridiagonalLayout}) = getindex_value(diagonaldata(M.A)) * M.B
copy(M::Rmul{<:SymTridiagonalLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A * getindex_value(diagonaldata(M.B))
copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout},<:SymTridiagonalLayout}) = getindex_value(diagonaldata(M.A)) * M.B


copy(M::Rmul{<:BidiagonalLayout,DiagonalLayout{OnesLayout}}) = _copy_oftype(M.A, eltype(M))
copy(M::Lmul{DiagonalLayout{OnesLayout},<:BidiagonalLayout}) =  _copy_oftype(M.B, eltype(M))
copy(M::Rmul{<:TridiagonalLayout,DiagonalLayout{OnesLayout}}) = _copy_oftype(M.A, eltype(M))
copy(M::Lmul{DiagonalLayout{OnesLayout},<:TridiagonalLayout}) =  _copy_oftype(M.B, eltype(M))
copy(M::Rmul{<:SymTridiagonalLayout,DiagonalLayout{OnesLayout}}) = _copy_oftype(M.A, eltype(M))
copy(M::Lmul{DiagonalLayout{OnesLayout},<:SymTridiagonalLayout}) =  _copy_oftype(M.B, eltype(M))
