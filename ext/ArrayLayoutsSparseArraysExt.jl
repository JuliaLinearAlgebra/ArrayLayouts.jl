module ArrayLayoutsSparseArraysExt

using ArrayLayouts
using ArrayLayouts: _copyto!, Factorization
using SparseArrays
# Specifying the full namespace is necessary because of https://github.com/JuliaLang/julia/issues/48533
# See https://github.com/JuliaStats/LogExpFunctions.jl/pull/63
import ArrayLayouts.LinearAlgebra

import Base: copyto!, \, /

# ambiguity from sparsematrix.jl
copyto!(dest::LayoutMatrix, src::SparseArrays.AbstractSparseMatrixCSC) =
    _copyto!(dest, src)

copyto!(dest::SubArray{<:Any,2,<:LayoutMatrix}, src::SparseArrays.AbstractSparseMatrixCSC) =
    _copyto!(dest, src)

@inline LinearAlgebra.dot(a::LayoutArray{<:Number}, b::SparseArrays.SparseVectorUnion{<:Number}) =
    ArrayLayouts.dot(a,b)

@inline LinearAlgebra.dot(a::SparseArrays.SparseVectorUnion{<:Number}, b::LayoutArray{<:Number}) =
    ArrayLayouts.dot(a,b)

# disambiguiate sparse matrix dispatches
macro _layoutldivsp(Typ)
    ret = quote
        (\)(x::SparseArrays.AbstractSparseMatrixCSC, A::$Typ; kwds...) = ArrayLayouts.ldiv(x,A; kwds...)
        (/)(x::SparseArrays.AbstractSparseMatrixCSC, A::$Typ; kwds...) = ArrayLayouts.ldiv(x,A; kwds...)
    end
    esc(ret)
end

macro layoutldivsp(Typ)
    esc(quote
        ArrayLayoutsSparseArraysExt.@_layoutldivsp $Typ
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UpperTriangular{T, <:$Typ{T}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UnitUpperTriangular{T, <:$Typ{T}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.LowerTriangular{T, <:$Typ{T}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UnitLowerTriangular{T, <:$Typ{T}} where T

        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UpperTriangular{T, <:SubArray{T,2,<:$Typ{T}}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UnitUpperTriangular{T, <:SubArray{T,2,<:$Typ{T}}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.LowerTriangular{T, <:SubArray{T,2,<:$Typ{T}}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UnitLowerTriangular{T, <:SubArray{T,2,<:$Typ{T}}} where T

        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UpperTriangular{T, <:LinearAlgebra.Adjoint{T,<:$Typ{T}}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UnitUpperTriangular{T, <:LinearAlgebra.Adjoint{T,<:$Typ{T}}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.LowerTriangular{T, <:LinearAlgebra.Adjoint{T,<:$Typ{T}}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UnitLowerTriangular{T, <:LinearAlgebra.Adjoint{T,<:$Typ{T}}} where T

        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UpperTriangular{T, <:LinearAlgebra.Transpose{T,<:$Typ{T}}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UnitUpperTriangular{T, <:LinearAlgebra.Transpose{T,<:$Typ{T}}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.LowerTriangular{T, <:LinearAlgebra.Transpose{T,<:$Typ{T}}} where T
        ArrayLayoutsSparseArraysExt.@_layoutldivsp LinearAlgebra.UnitLowerTriangular{T, <:LinearAlgebra.Transpose{T,<:$Typ{T}}} where T
    end)
end

@_layoutldivsp LayoutVector

macro layoutmatrixsp(Typ)
    esc(quote
        ArrayLayoutsSparseArraysExt.@layoutldivsp $Typ
    end)
end

@layoutmatrixsp LayoutMatrix

end
