module ArrayLayouts
using Base, Base.Broadcast, LinearAlgebra, FillArrays
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
      all, any, isbitsunion, issubset, replace_in_print_matrix, replace_with_centered_mark

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, Broadcasted, broadcasted,
                        combine_eltypes, DefaultArrayStyle, instantiate, materialize,
                        materialize!, eltypes

import LinearAlgebra: AbstractTriangular, AbstractQ, checksquare, pinv, fill!, tilebufsize, Abuf, Bbuf, Cbuf, dot, factorize, qr, lu, cholesky, 
                        norm2, norm1, normInf, normMinusInf

import LinearAlgebra.BLAS: BlasFloat, BlasReal, BlasComplex

import FillArrays: AbstractFill, getindex_value

if VERSION < v"1.2-"
    import Base: has_offset_axes
    require_one_based_indexing(A...) = !has_offset_axes(A...) || throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
else
    import Base: require_one_based_indexing    
end     

export materialize, materialize!, MulAdd, muladd!, Ldiv, Lmul, Rmul, MemoryLayout, AbstractStridedLayout,
        DenseColumnMajor, ColumnMajor, ZerosLayout, FillLayout, AbstractColumnMajor, RowMajor, AbstractRowMajor,
        DiagonalLayout, ScalarLayout, SymTridiagonalLayout, HermitianLayout, SymmetricLayout, TriangularLayout, 
        UnknownLayout, AbstractBandedLayout, ApplyBroadcastStyle, ConjLayout, AbstractFillLayout,
        colsupport, rowsupport, lazy_getindex

struct ApplyBroadcastStyle <: BroadcastStyle end
@inline function copyto!(dest::AbstractArray, bc::Broadcasted{ApplyBroadcastStyle}) 
    @assert length(bc.args) == 1
    copyto!(dest, first(bc.args))
end

include("memorylayout.jl")
include("muladd.jl")
include("lmul.jl")
include("ldiv.jl")
include("diagonal.jl")
include("triangular.jl")

@inline sub_materialize(_, V) = Array(V)
@inline sub_materialize(V::SubArray) = sub_materialize(MemoryLayout(typeof(V)), V)

@inline lazy_getindex(A, I...) = sub_materialize(view(A, I...))


end