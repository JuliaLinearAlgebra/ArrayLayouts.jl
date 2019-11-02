



struct Ldiv{StyleA, StyleB, AType, BType}
    A::AType
    B::BType
end

Ldiv{StyleA, StyleB}(A::AType, B::BType) where {StyleA,StyleB,AType,BType} = 
    Ldiv{StyleA,StyleB,AType,BType}(A,B)

Ldiv(A::AType, B::BType) where {AType,BType} = 
    Ldiv{typeof(MemoryLayout(AType)),typeof(MemoryLayout(BType)),AType,BType}(A, B)

struct LdivBroadcastStyle <: BroadcastStyle end

size(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractMatrix}) = (size(L.A, 2),size(L.B,2))
size(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) = (size(L.A, 2),)
axes(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractMatrix}) = (axes(L.A, 2),axes(L.B,2))
axes(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) = (axes(L.A, 2),)    
length(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) =size(L.A, 2)

_ldivaxes(::Tuple{}, ::Tuple{}) = ()
_ldivaxes(::Tuple{}, Bax::Tuple) = Bax
_ldivaxes(::Tuple{<:Any}, ::Tuple{<:Any}) = ()
_ldivaxes(::Tuple{<:Any}, Bax::Tuple{<:Any,<:Any}) = (OneTo(1),last(Bax))
_ldivaxes(Aax::Tuple{<:Any,<:Any}, ::Tuple{<:Any}) = (last(Aax),)
_ldivaxes(Aax::Tuple{<:Any,<:Any}, Bax::Tuple{<:Any,<:Any}) = (last(Aax),last(Bax))

@inline ldivaxes(A, B) = _ldivaxes(axes(A), axes(B))

ndims(L::Ldiv) = ndims(last(L.args))
eltype(M::Ldiv) = promote_type(Base.promote_op(inv, eltype(M.A)), eltype(M.B))

BroadcastStyle(::Type{<:Ldiv}) = ApplyBroadcastStyle()
broadcastable(M::Ldiv) = M

similar(A::Ldiv, ::Type{T}) where T = similar(Array{T}, axes(A))
similar(A::Ldiv) = similar(A, eltype(A))

function instantiate(L::Ldiv)
    check_ldiv_axes(L.A, L.B)
    Ldiv(instantiate(L.A), instantiate(L.B))
end


check_ldiv_axes(A, B) =
    axes(A,1) == axes(B,1) || throw(DimensionMismatch("First axis of A, $(axes(A,1)), and first axis of B, $(axes(B,1)) must match"))



copy(M::Ldiv) = copyto!(similar(M), M)
materialize(M::Ldiv) = copy(instantiate(M))

_ldiv!(A, B) = ldiv!(factorize(A), B)
_ldiv!(A::Factorization, B) = ldiv!(A, B)

_ldiv!(dest, A, B) = ldiv!(dest, factorize(A), B)
_ldiv!(dest, A::Factorization, B) = ldiv!(dest, A, B)


materialize!(M::Ldiv) = _ldiv!(M.A, M.B)
if VERSION ≥ v"1.1-pre"
    copyto!(dest::AbstractArray, M::Ldiv) = _ldiv!(dest, M.A, M.B)
else
    copyto!(dest::AbstractArray, M::Ldiv) = _ldiv!(dest, M.A, copy(M.B))
end

const MatLdivVec{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractVector{V}}
const MatLdivMat{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractMatrix{V}}
const BlasMatLdivVec{styleA, styleB, T<:BlasFloat} = MatLdivVec{styleA, styleB, T, T}
const BlasMatLdivMat{styleA, styleB, T<:BlasFloat} = MatLdivMat{styleA, styleB, T, T}


macro lazyldiv(Typ)
    esc(quote
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractVector) = ArrayLayouts.materialize!(ArrayLayouts.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractMatrix) = ArrayLayouts.materialize!(ArrayLayouts.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedVector) = ArrayLayouts.materialize!(ArrayLayouts.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedMatrix) = ArrayLayouts.materialize!(ArrayLayouts.Ldiv(A,x))

        Base.:\(A::$Typ, x::AbstractVector) = ArrayLayouts.materialize(ArrayLayouts.Ldiv(A,x))
        Base.:\(A::$Typ, x::AbstractMatrix) = ArrayLayouts.materialize(ArrayLayouts.Ldiv(A,x))
    end)
end