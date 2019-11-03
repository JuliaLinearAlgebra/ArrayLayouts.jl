struct QLayout <: MemoryLayout end

MemoryLayout(::Type{<:AbstractQ}) = QLayout()

transposelayout(::QLayout) = QLayout()


copy(M::Lmul{QLayout}) = copyto!(similar(M), M)

function copyto!(dest::AbstractArray{T}, M::Lmul{QLayout}) where T
    A,B = M.A,M.B
    if size(dest,1) == size(B,1) 
        copyto!(dest, B)
    else
        copyto!(view(dest,1:size(B,1),:), B)
        zero!(@view(dest[size(B,1)+1:end,:]))
    end
    materialize!(Lmul(A,dest))
end

function copyto!(dest::AbstractArray, M::Ldiv{QLayout})
    A,B = M.A,M.B
    copyto!(dest, B)
    materialize!(Ldiv(A,dest))
end

materialize!(M::Ldiv{QLayout}) = materialize!(Lmul(M.A',M.B))
