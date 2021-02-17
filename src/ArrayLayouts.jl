module ArrayLayouts
using Base, Base.Broadcast, LinearAlgebra, FillArrays, SparseArrays, Compat
import LinearAlgebra.BLAS

import Base: AbstractArray, AbstractMatrix, AbstractVector,
        ReinterpretArray, ReshapedArray, AbstractCartesianIndex, Slice,
             RangeIndex, BroadcastStyle, copyto!, length, broadcastable, axes,
             getindex, eltype, tail, IndexStyle, IndexLinear, getproperty,
             *, +, -, /, \, ==, isinf, isfinite, sign, angle, show, isless,
         fld, cld, div, min, max, minimum, maximum, mod,
         <, ≤, >, ≥, promote_rule, convert, copy,
         size, step, isempty, length, first, last, ndims,
         getindex, setindex!, intersect, @_inline_meta, inv,
         sort, sort!, issorted, sortperm, diff, cumsum, sum, in, broadcast,
         eltype, parent, real, imag,
         conj, transpose, adjoint, permutedims, vec,
         exp, log, sqrt, cos, sin, tan, csc, sec, cot,
                   cosh, sinh, tanh, csch, sech, coth,
                   acos, asin, atan, acsc, asec, acot,
                   acosh, asinh, atanh, acsch, asech, acoth, (:),
         AbstractMatrix, AbstractArray, checkindex, unsafe_length, OneTo, one, zero,
        to_shape, _sub2ind, print_matrix, print_matrix_row, print_matrix_vdots,
      checkindex, Slice, @propagate_inbounds, @_propagate_inbounds_meta,
      _in_range, _range, _rangestyle, Ordered,
      ArithmeticWraps, floatrange, reverse, unitrange_last,
      AbstractArray, AbstractVector, axes, (:), _sub2ind_recurse, broadcast, promote_eltypeof,
      similar, @_gc_preserve_end, @_gc_preserve_begin,
      @nexprs, @ncall, @ntuple, tuple_type_tail,
      all, any, isbitsunion, issubset, replace_in_print_matrix, replace_with_centered_mark,
      strides, unsafe_convert, first_index, unalias, union

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, Broadcasted, broadcasted,
                        combine_eltypes, DefaultArrayStyle, instantiate, materialize,
                        materialize!, eltypes

import LinearAlgebra: AbstractTriangular, AbstractQ, checksquare, pinv, fill!, tilebufsize, Abuf, Bbuf, Cbuf, factorize, qr, lu, cholesky,
                        norm2, norm1, normInf, normMinusInf, qr, lu, qr!, lu!, AdjOrTrans, HermOrSym, copy_oftype,
                        AdjointAbsVec, TransposeAbsVec

import LinearAlgebra.BLAS: BlasFloat, BlasReal, BlasComplex

import FillArrays: AbstractFill, getindex_value, axes_print_matrix_row

import Base: require_one_based_indexing

export materialize, materialize!, MulAdd, muladd!, Ldiv, Rdiv, Lmul, Rmul, Dot,
        lmul, rmul, mul, ldiv, rdiv, mul, MemoryLayout, AbstractStridedLayout,
        DenseColumnMajor, ColumnMajor, ZerosLayout, FillLayout, AbstractColumnMajor, RowMajor, AbstractRowMajor, UnitStride,
        DiagonalLayout, ScalarLayout, SymTridiagonalLayout, TridiagonalLayout, BidiagonalLayout,
        HermitianLayout, SymmetricLayout, TriangularLayout,
        UnknownLayout, AbstractBandedLayout, ApplyBroadcastStyle, ConjLayout, AbstractFillLayout, DualLayout,
        colsupport, rowsupport, layout_getindex, QLayout, LayoutArray, LayoutMatrix, LayoutVector,
        RangeCumsum

struct ApplyBroadcastStyle <: BroadcastStyle end
@inline function copyto!(dest::AbstractArray, bc::Broadcasted{ApplyBroadcastStyle})
    @assert length(bc.args) == 1
    copyto!(dest, first(bc.args))
end

# Subtypes of LayoutArray default to
# ArrayLayouts routines
abstract type LayoutArray{T,N} <: AbstractArray{T,N} end
const LayoutMatrix{T} = LayoutArray{T,2}
const LayoutVector{T} = LayoutArray{T,1}

## TODO: Following are type piracy whch may be removed in Julia v1.5
_transpose_strides(a) = (a,1)
_transpose_strides(a,b) = (b,a)
strides(A::Adjoint) = _transpose_strides(strides(parent(A))...)
strides(A::Transpose) = _transpose_strides(strides(parent(A))...)

"""
    ConjPtr{T}

represents that the entry is the complex-conjugate of the pointed to entry.
"""
struct ConjPtr{T}
    ptr::Ptr{T}
end


# work-around issue with complex conjugation of pointer
unsafe_convert(::Type{Ptr{T}}, Ac::Adjoint{<:Complex}) where T<:Complex = unsafe_convert(ConjPtr{T}, parent(Ac))
unsafe_convert(::Type{ConjPtr{T}}, Ac::Adjoint{<:Complex}) where T<:Complex = unsafe_convert(Ptr{T}, parent(Ac))
function unsafe_convert(::Type{ConjPtr{T}}, V::SubArray{T,2}) where {T,N,P}
    kr, jr = parentindices(V)
    unsafe_convert(Ptr{T}, view(parent(V)', jr, kr))
end

include("memorylayout.jl")
include("mul.jl")
include("muladd.jl")
include("lmul.jl")
include("ldiv.jl")
include("diagonal.jl")
include("triangular.jl")
include("factorizations.jl")

@inline sub_materialize(_, V, _) = Array(V)
@inline sub_materialize(L, V) = sub_materialize(L, V, axes(V))
@inline sub_materialize(V::SubArray) = sub_materialize(MemoryLayout(V), V)
@inline sub_materialize(V::AbstractArray) = V # Anything not a SubArray is already materialized

@inline layout_getindex(A, I...) = sub_materialize(view(A, I...))

macro _layoutgetindex(Typ)
    esc(quote
        @inline Base.getindex(A::$Typ, kr::Colon, jr::Colon) = ArrayLayouts.layout_getindex(A, kr, jr)
        @inline Base.getindex(A::$Typ, kr::Colon, jr::AbstractUnitRange) = ArrayLayouts.layout_getindex(A, kr, jr)
        @inline Base.getindex(A::$Typ, kr::AbstractUnitRange, jr::Colon) = ArrayLayouts.layout_getindex(A, kr, jr)
        @inline Base.getindex(A::$Typ, kr::AbstractUnitRange, jr::AbstractUnitRange) = ArrayLayouts.layout_getindex(A, kr, jr)
        @inline Base.getindex(A::$Typ, kr::AbstractVector, jr::AbstractVector) = ArrayLayouts.layout_getindex(A, kr, jr)
        @inline Base.getindex(A::$Typ, kr::Colon, jr::AbstractVector) = ArrayLayouts.layout_getindex(A, kr, jr)
        @inline Base.getindex(A::$Typ, kr::Colon, jr::Integer) = ArrayLayouts.layout_getindex(A, kr, jr)
        @inline Base.getindex(A::$Typ, kr::AbstractVector, jr::Colon) = ArrayLayouts.layout_getindex(A, kr, jr)
        @inline Base.getindex(A::$Typ, kr::Integer, jr::Colon) = ArrayLayouts.layout_getindex(A, kr, jr)
        @inline Base.getindex(A::$Typ, kr::Integer, jr::AbstractVector) = ArrayLayouts.layout_getindex(A, kr, jr)
    end)
end

macro layoutgetindex(Typ)
    esc(quote
        ArrayLayouts.@_layoutgetindex $Typ
        ArrayLayouts.@_layoutgetindex LinearAlgebra.AbstractTriangular{<:Any,<:$Typ}
        ArrayLayouts.@_layoutgetindex LinearAlgebra.Symmetric{<:Any,<:$Typ}
        ArrayLayouts.@_layoutgetindex LinearAlgebra.Hermitian{<:Any,<:$Typ}
        ArrayLayouts.@_layoutgetindex LinearAlgebra.Adjoint{<:Any,<:$Typ}
        ArrayLayouts.@_layoutgetindex LinearAlgebra.Transpose{<:Any,<:$Typ}
        ArrayLayouts.@_layoutgetindex LinearAlgebra.SubArray{<:Any,2,<:$Typ}
    end)
end


macro layoutmatrix(Typ)
    esc(quote
        ArrayLayouts.@layoutldiv $Typ
        ArrayLayouts.@layoutmul $Typ
        ArrayLayouts.@layoutlmul $Typ
        ArrayLayouts.@layoutfactorizations $Typ
        ArrayLayouts.@layoutgetindex $Typ
    end)
end

@layoutmatrix LayoutMatrix

for Typ in (:LayoutArray, :(Transpose{<:Any,<:LayoutMatrix}), :(Adjoint{<:Any,<:LayoutMatrix}), :(Symmetric{<:Any,<:LayoutMatrix}), :(Hermitian{<:Any,<:LayoutMatrix}))
    @eval begin
        LinearAlgebra.lmul!(α::Number, A::$Typ) = lmul!(α, A)
        LinearAlgebra.rmul!(A::$Typ, α::Number) = rmul!(A, α)
        LinearAlgebra.ldiv!(α::Number, A::$Typ) = ldiv!(α, A)
        LinearAlgebra.rdiv!(A::$Typ, α::Number) = rdiv!(A, α)
    end
end

getindex(A::LayoutVector, kr::AbstractVector) = layout_getindex(A, kr)
getindex(A::LayoutVector, kr::Colon) = layout_getindex(A, kr)
getindex(A::AdjOrTrans{<:Any,<:LayoutVector}, kr::Integer, jr::Colon) = layout_getindex(A, kr, jr)
getindex(A::AdjOrTrans{<:Any,<:LayoutVector}, kr::Integer, jr::AbstractVector) = layout_getindex(A, kr, jr)

*(A::Diagonal{<:Any,<:LayoutVector}, B::Diagonal{<:Any,<:LayoutVector}) = mul(A, B)
*(A::Diagonal{<:Any,<:LayoutVector}, B::AbstractMatrix) = mul(A, B)
*(A::AbstractMatrix, B::Diagonal{<:Any,<:LayoutVector}) = mul(A, B)
*(A::Diagonal{<:Any,<:LayoutVector}, B::LayoutMatrix) = mul(A, B)
*(A::LayoutMatrix, B::Diagonal{<:Any,<:LayoutVector}) = mul(A, B)
*(A::Diagonal{<:Any,<:LayoutVector}, B::Diagonal) = mul(A, B)
*(A::Diagonal, B::Diagonal{<:Any,<:LayoutVector}) = mul(A, B)

*(A::LayoutMatrix, B::Adjoint{<:Any,<:AbstractTriangular}) = mul(A, B)
*(A::LayoutMatrix, B::Transpose{<:Any,<:AbstractTriangular}) = mul(A, B)
*(A::Adjoint{<:Any,<:AbstractTriangular}, B::LayoutMatrix) = mul(A, B)
*(A::Transpose{<:Any,<:AbstractTriangular}, B::LayoutMatrix) = mul(A, B)

for Mod in (:Adjoint, :Transpose, :Symmetric, :Hermitian)
    @eval begin
        *(A::Diagonal{<:Any,<:LayoutVector}, B::$Mod{<:Any,<:LayoutMatrix}) = mul(A,B)
        *(A::$Mod{<:Any,<:LayoutMatrix}, B::Diagonal{<:Any,<:LayoutVector}) = mul(A,B)
    end
end
\(A::Diagonal{<:Any,<:LayoutVector}, B::Diagonal{<:Any,<:LayoutVector}) = ldiv(A, B)
\(A::Diagonal{<:Any,<:LayoutVector}, B::AbstractMatrix) = ldiv(A, B)
\(A::AbstractMatrix, B::Diagonal{<:Any,<:LayoutVector}) = ldiv(A, B)
\(A::Diagonal{<:Any,<:LayoutVector}, B::LayoutMatrix) = ldiv(A, B)
\(A::LayoutMatrix, B::Diagonal{<:Any,<:LayoutVector}) = ldiv(A, B)
\(A::Diagonal{<:Any,<:LayoutVector}, B::Diagonal) = ldiv(A, B)
\(A::Diagonal, B::Diagonal{<:Any,<:LayoutVector}) = ldiv(A, B)


_copyto!(_, _, dest::AbstractArray{T,N}, src::AbstractArray{V,N}) where {T,V,N} =
    Base.invoke(copyto!, Tuple{AbstractArray{T,N},AbstractArray{V,N}}, dest, src)


_copyto!(dest, src) = _copyto!(MemoryLayout(dest), MemoryLayout(src), dest, src)
copyto!(dest::LayoutArray{<:Any,N}, src::LayoutArray{<:Any,N}) where N = _copyto!(dest, src)
copyto!(dest::AbstractArray{<:Any,N}, src::LayoutArray{<:Any,N}) where N = _copyto!(dest, src)
copyto!(dest::LayoutArray{<:Any,N}, src::AbstractArray{<:Any,N}) where N = _copyto!(dest, src)

copyto!(dest::SubArray{<:Any,N,<:LayoutArray}, src::SubArray{<:Any,N,<:LayoutArray}) where N = _copyto!(dest, src)
copyto!(dest::SubArray{<:Any,N,<:LayoutArray}, src::LayoutArray{<:Any,N}) where N = _copyto!(dest, src)
copyto!(dest::LayoutArray{<:Any,N}, src::SubArray{<:Any,N,<:LayoutArray}) where N = _copyto!(dest, src)
copyto!(dest::SubArray{<:Any,N,<:LayoutArray}, src::AbstractArray{<:Any,N}) where N = _copyto!(dest, src)
copyto!(dest::AbstractArray{<:Any,N}, src::SubArray{<:Any,N,<:LayoutArray}) where N = _copyto!(dest, src)

copyto!(dest::LayoutMatrix, src::AdjOrTrans{<:Any,<:LayoutArray}) = _copyto!(dest, src)
copyto!(dest::LayoutMatrix, src::SubArray{<:Any,2,<:AdjOrTrans{<:Any,<:LayoutArray}}) = _copyto!(dest, src)
copyto!(dest::AbstractMatrix, src::AdjOrTrans{<:Any,<:LayoutArray}) = _copyto!(dest, src)
copyto!(dest::SubArray{<:Any,2,<:LayoutArray}, src::AdjOrTrans{<:Any,<:LayoutArray}) = _copyto!(dest, src)
copyto!(dest::SubArray{<:Any,2,<:LayoutMatrix}, src::SubArray{<:Any,2,<:AdjOrTrans{<:Any,<:LayoutArray}}) = _copyto!(dest, src)
copyto!(dest::AbstractMatrix, src::SubArray{<:Any,2,<:AdjOrTrans{<:Any,<:LayoutArray}}) = _copyto!(dest, src)
# ambiguity from sparsematrix.jl
copyto!(dest::LayoutMatrix, src::SparseArrays.AbstractSparseMatrixCSC) = _copyto!(dest, src)
copyto!(dest::SubArray{<:Any,2,<:LayoutMatrix}, src::SparseArrays.AbstractSparseMatrixCSC) = _copyto!(dest, src)

# avoid bad copy in Base
Base.map(::typeof(copy), D::Diagonal{<:LayoutArray}) = Diagonal(map(copy, D.diag))
Base.permutedims(D::Diagonal{<:Any,<:LayoutVector}) = D


zero!(A::AbstractArray{T}) where T = fill!(A,zero(T))
function zero!(A::AbstractArray{<:AbstractArray})
    for a in A
        zero!(a)
    end
    A
end

_fill_lmul!(β, A::AbstractArray{T}) where T = iszero(β) ? zero!(A) : lmul!(β, A)


# Elementary reflection similar to LAPACK. The reflector is not Hermitian but
# ensures that tridiagonalization of Hermitian matrices become real. See lawn72
@inline function reflector!(x::AbstractVector)
    require_one_based_indexing(x)
    n = length(x)
    n == 0 && return zero(eltype(x))
    @inbounds begin
        ξ1 = x[1]
        normu = abs2(ξ1)
        for i = 2:n
            normu += abs2(x[i])
        end
        if iszero(normu)
            return zero(ξ1/normu)
        end
        normu = sqrt(normu)
        ν = copysign(normu, real(ξ1))
        ξ1 += ν
        x[1] = -ν
        for i = 2:n
            x[i] /= ξ1
        end
    end
    ξ1/ν
end

# apply reflector from left
@inline function reflectorApply!(x::AbstractVector, τ::Number, A::AbstractVecOrMat)
    m,n = size(A,1),size(A,2)
    if length(x) != m
        throw(DimensionMismatch("reflector has length $(length(x)), which must match the first dimension of matrix A, $m"))
    end
    m == 0 && return A
    @inbounds begin
        for j = 1:n
            # dot
            vAj = A[1, j]
            for i = 2:m
                vAj += x[i]'*A[i, j]
            end

            vAj = conj(τ)*vAj

            # ger
            A[1, j] -= vAj
            for i = 2:m
                A[i, j] -= x[i]*vAj
            end
        end
    end
    return A
end

###
# printing
###

layout_replace_in_print_matrix(_, A, i, j, s) =
    i in colsupport(A,j) ? s : Base.replace_with_centered_mark(s)

Base.replace_in_print_matrix(A::Union{LayoutVector,
                                        LayoutMatrix,
                                      UpperTriangular{<:Any,<:LayoutMatrix},
                                      UnitUpperTriangular{<:Any,<:LayoutMatrix},
                                      LowerTriangular{<:Any,<:LayoutMatrix},
                                      UnitLowerTriangular{<:Any,<:LayoutMatrix},
                                      AdjOrTrans{<:Any,<:LayoutMatrix},
                                      HermOrSym{<:Any,<:LayoutMatrix},
                                      SubArray{<:Any,2,<:LayoutMatrix}}, i::Integer, j::Integer, s::AbstractString) =
    layout_replace_in_print_matrix(MemoryLayout(A), A, i, j, s)

Base.print_matrix_row(io::IO,
        X::Union{LayoutMatrix,
        LayoutVector,
        AbstractTriangular{<:Any,<:LayoutMatrix},
        AdjOrTrans{<:Any,<:LayoutMatrix},
        AdjOrTrans{<:Any,<:LayoutVector},
        HermOrSym{<:Any,<:LayoutMatrix},
        SubArray{<:Any,2,<:LayoutMatrix},
        Diagonal{<:Any,<:LayoutVector}}, A::Vector,
        i::Integer, cols::AbstractVector, sep::AbstractString) =
        axes_print_matrix_row(axes(X), io, X, A, i, cols, sep)


include("cumsum.jl")

end
