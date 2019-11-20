for Typ in (:Ldiv, :Rdiv)
    @eval begin
        struct $Typ{StyleA, StyleB, AType, BType}
            A::AType
            B::BType
        end

        @inline $Typ{StyleA, StyleB}(A::AType, B::BType) where {StyleA,StyleB,AType,BType} =
            $Typ{StyleA,StyleB,AType,BType}(A,B)

        @inline $Typ(A::AType, B::BType) where {AType,BType} =
            $Typ{typeof(MemoryLayout(AType)),typeof(MemoryLayout(BType)),AType,BType}(A, B)

        @inline BroadcastStyle(::Type{<:$Typ}) = ApplyBroadcastStyle()
        @inline broadcastable(M::$Typ) = M

        similar(A::$Typ, ::Type{T}, axes) where T = similar(Array{T}, axes)
        similar(A::$Typ, ::Type{T}) where T = similar(A, T, axes(A))
        similar(A::$Typ) = similar(A, eltype(A))

        @inline copy(M::$Typ) = copyto!(similar(M), M)
        @inline materialize(M::$Typ) = copy(instantiate(M))
    end
end

@inline _ldivaxes(::Tuple{}, ::Tuple{}) = ()
@inline _ldivaxes(::Tuple{}, Bax::Tuple) = Bax
@inline _ldivaxes(::Tuple{<:Any}, ::Tuple{<:Any}) = ()
@inline _ldivaxes(::Tuple{<:Any}, Bax::Tuple{<:Any,<:Any}) = (OneTo(1),last(Bax))
@inline _ldivaxes(Aax::Tuple{<:Any,<:Any}, ::Tuple{<:Any}) = (last(Aax),)
@inline _ldivaxes(Aax::Tuple{<:Any,<:Any}, Bax::Tuple{<:Any,<:Any}) = (last(Aax),last(Bax))

@inline ldivaxes(A, B) = _ldivaxes(axes(A), axes(B))

@inline axes(L::Ldiv) = ldivaxes(L.A, L.B)
@inline size(L::Ldiv) = map(length, axes(L))
@inline length(L::Ldiv{<:Any,<:Any,<:Any,<:AbstractVector}) =size(L.A, 2)

@inline size(L::Rdiv) = (size(L.A, 1),size(L.B,1))
@inline axes(L::Rdiv) = (axes(L.A, 1),axes(L.B,1))

@inline ndims(L::Ldiv) = ndims(L.B)
@inline ndims(L::Rdiv) = 2
@inline eltype(M::Ldiv) = promote_type(Base.promote_op(inv, eltype(M.A)), eltype(M.B))
@inline eltype(M::Rdiv) = promote_type(eltype(M.A), Base.promote_op(inv, eltype(M.B)))


check_ldiv_axes(A, B) =
    axes(A,1) == axes(B,1) || throw(DimensionMismatch("First axis of A, $(axes(A,1)), and first axis of B, $(axes(B,1)) must match"))

check_rdiv_axes(A, B) =
    axes(A,2) == axes(B,2) || throw(DimensionMismatch("Second axis of A, $(axes(A,2)), and second axis of B, $(axes(B,2)) must match"))



@inline function instantiate(L::Ldiv)
    check_ldiv_axes(L.A, L.B)
    Ldiv(instantiate(L.A), instantiate(L.B))
end

@inline function instantiate(L::Rdiv)
    check_rdiv_axes(L.A, L.B)
    Rdiv(instantiate(L.A), instantiate(L.B))
end

@inline _ldiv!(A, B) = ldiv!(factorize(A), B)
@inline _ldiv!(A::Factorization, B) = ldiv!(A, B)

@inline _ldiv!(dest, A, B) = ldiv!(dest, factorize(A), B)
@inline _ldiv!(dest, A::Factorization, B) = ldiv!(dest, A, B)
@inline _ldiv!(dest, A::Transpose{<:Any,<:Factorization}, B) = ldiv!(dest, A, B)
@inline _ldiv!(dest, A::Adjoint{<:Any,<:Factorization}, B) = ldiv!(dest, A, B)

@inline ldiv(A, B) = materialize(Ldiv(A,B))
@inline rdiv(A, B) = materialize(Rdiv(A,B))

@inline materialize!(M::Ldiv) = _ldiv!(M.A, M.B)
@inline materialize!(M::Rdiv) = materialize!(Lmul(M.B', M.A'))'
@inline copyto!(dest::AbstractArray, M::Rdiv) = copyto!(dest', Ldiv(M.B', M.A'))'

if VERSION ≥ v"1.1-pre"
    @inline copyto!(dest::AbstractArray, M::Ldiv) = _ldiv!(dest, M.A, M.B)
else
    @inline copyto!(dest::AbstractArray, M::Ldiv) = _ldiv!(dest, M.A, copy(M.B))
end

const MatLdivVec{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractVector{V}}
const MatLdivMat{styleA, styleB, T, V} = Ldiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractMatrix{V}}
const BlasMatLdivVec{styleA, styleB, T<:BlasFloat} = MatLdivVec{styleA, styleB, T, T}
const BlasMatLdivMat{styleA, styleB, T<:BlasFloat} = MatLdivMat{styleA, styleB, T, T}

const MatRdivMat{styleA, styleB, T, V} = Rdiv{styleA, styleB, <:AbstractMatrix{T}, <:AbstractMatrix{V}}
const BlasMatRdivMat{styleA, styleB, T<:BlasFloat} = MatRdivMat{styleA, styleB, T, T}

# function materialize!(L::BlasMatLdivVec{<:AbstractColumnMajor,<:AbstractColumnMajor})

# end


macro lazyldiv(Typ)
    esc(quote
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractVector) = ArrayLayouts.materialize!(ArrayLayouts.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::AbstractMatrix) = ArrayLayouts.materialize!(ArrayLayouts.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedVector) = ArrayLayouts.materialize!(ArrayLayouts.Ldiv(A,x))
        LinearAlgebra.ldiv!(A::$Typ, x::StridedMatrix) = ArrayLayouts.materialize!(ArrayLayouts.Ldiv(A,x))

        Base.:\(A::$Typ, x::AbstractVector) = ArrayLayouts.materialize(ArrayLayouts.Ldiv(A,x))
        Base.:\(A::$Typ, x::AbstractMatrix) = ArrayLayouts.materialize(ArrayLayouts.Ldiv(A,x))

        Base.:\(x::AbstractMatrix, A::$Typ) = ArrayLayouts.materialize(ArrayLayouts.Ldiv(x,A))
        Base.:\(x::Diagonal, A::$Typ) = ArrayLayouts.materialize(ArrayLayouts.Ldiv(x,A))

        Base.:\(x::$Typ, A::$Typ) = ArrayLayouts.materialize(ArrayLayouts.Ldiv(x,A))

        Base.:/(A::$Typ, x::AbstractVector) = ArrayLayouts.materialize(ArrayLayouts.Rdiv(A,x))
        Base.:/(A::$Typ, x::AbstractMatrix) = ArrayLayouts.materialize(ArrayLayouts.Rdiv(A,x))

        Base.:/(x::AbstractMatrix, A::$Typ) = ArrayLayouts.materialize(ArrayLayouts.Rdiv(x,A))
        Base.:/(x::Diagonal, A::$Typ) = ArrayLayouts.materialize(ArrayLayouts.Rdiv(x,A))

        Base.:/(x::$Typ, A::$Typ) = ArrayLayouts.materialize(ArrayLayouts.Rdiv(x,A))
    end)
end
