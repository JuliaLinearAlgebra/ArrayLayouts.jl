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

_qr(layout, axes, A, args...; kwds...) = Base.invoke(qr, Tuple{AbstractMatrix{eltype(A)}}, A, args...; kwds...)
_qr!(layout, axes, A, args...; kwds...) = Base.invoke(qr!, Tuple{AbstractMatrix{eltype(A)}}, A, args...; kwds...)
_lu(layout, axes, A, args...; kwds...) = Base.invoke(lu, Tuple{AbstractMatrix{eltype(A)}}, A, args...; kwds...)
_lu!(layout, axes, A, args...; kwds...) = Base.invoke(lu!, Tuple{AbstractMatrix{eltype(A)}}, A, args...; kwds...)
_factorize(layout, axes, A) = Base.invoke(factorize, Tuple{AbstractMatrix{eltype(A)}}, A)

macro _layoutfactorizations(Typ)
    esc(quote
        LinearAlgebra.qr(A::$Typ, args...; kwds...) = ArrayLayouts._qr(ArrayLayouts.MemoryLayout(typeof(A)), axes(A), A, args...; kwds...)
        LinearAlgebra.qr!(A::$Typ, args...; kwds...) = ArrayLayouts._qr!(ArrayLayouts.MemoryLayout(typeof(A)), axes(A), A, args...; kwds...)
        LinearAlgebra.lu(A::$Typ, args...; kwds...) = ArrayLayouts._lu(ArrayLayouts.MemoryLayout(typeof(A)), axes(A), A, args...; kwds...)
        LinearAlgebra.lu!(A::$Typ, args...; kwds...) = ArrayLayouts._lu!(ArrayLayouts.MemoryLayout(typeof(A)), axes(A), A, args...; kwds...)
        LinearAlgebra.factorize(A::$Typ) = ArrayLayouts._factorize(ArrayLayouts.MemoryLayout(typeof(A)), axes(A), A)
    end)
end

macro layoutfactorizations(Typ)
    esc(quote
        ArrayLayouts.@_layoutfactorizations $Typ
        ArrayLayouts.@_layoutfactorizations SubArray{<:Any,2,<:$Typ}
    end)
end