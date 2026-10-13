module ArrayLayoutsSparseArraysExt

using ArrayLayouts
using ArrayLayouts: copyto!_layout, equals_layout
using SparseArrays
# Specifying the full namespace is necessary because of https://github.com/JuliaLang/julia/issues/48533
# See https://github.com/JuliaStats/LogExpFunctions.jl/pull/63
import ArrayLayouts.LinearAlgebra
using ArrayLayouts.LinearAlgebra: Adjoint, Transpose, Symmetric, Hermitian, Diagonal, UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular

import Base: copyto!, ==

# ambiguity from sparsematrix.jl
copyto!(dest::LayoutMatrix, src::SparseArrays.AbstractSparseMatrixCSC) =
	copyto!_layout(dest, src)

copyto!(dest::SubArray{<:Any,2,<:LayoutMatrix}, src::SparseArrays.AbstractSparseMatrixCSC) =
	copyto!_layout(dest, src)

@inline LinearAlgebra.dot(a::LayoutArray{<:Number}, b::SparseArrays.SparseVectorUnion{<:Number}) =
	ArrayLayouts.dot(a,b)

@inline LinearAlgebra.dot(a::SparseArrays.SparseVectorUnion{<:Number}, b::LayoutArray{<:Number}) =
	ArrayLayouts.dot(a,b)


# ambiguity between \ for sparse matrices and the layout overloads of \ (see @layoutldiv).
# SparseArrays' solvers need strided right-hand sides, so convert to Array.
# Symmetric/Hermitian are included since CHOLMOD does not support ldiv! with other arrays.
const LayoutLdivRHS = Any[:LayoutVector, :LayoutMatrix]
for Tri in (:UpperTriangular, :UnitUpperTriangular, :LowerTriangular, :UnitLowerTriangular),
		Par in (:(LayoutMatrix{T}), :(SubArray{T,2,<:LayoutMatrix{T}}), :(Adjoint{T,<:LayoutMatrix{T}}), :(Transpose{T,<:LayoutMatrix{T}}))
	push!(LayoutLdivRHS, :($Tri{T,<:$Par} where T))
end
for Sp in (:(SparseArrays.AbstractSparseMatrixCSC),
			:(Adjoint{<:Any,<:SparseArrays.AbstractSparseMatrixCSC}), :(Transpose{<:Any,<:SparseArrays.AbstractSparseMatrixCSC}),
			:(Symmetric{<:Any,<:SparseArrays.AbstractSparseMatrixCSC}), :(Hermitian{<:Any,<:SparseArrays.AbstractSparseMatrixCSC})),
		Typ in LayoutLdivRHS
	@eval Base.:\(A::$Sp, B::$Typ) = A \ convert(Array, B)
	Typ === :LayoutVector && @eval Base.:\(A::$Sp, B::Diagonal{<:Any,<:LayoutVector}) = A \ Diagonal(convert(Array, B.diag))
end

# ambiguity from SparseArrays.ReadOnly
if isdefined(SparseArrays, :ReadOnly)
	==(A::SparseArrays.ReadOnly, B::LayoutVector) = equals_layout(A, B)
	==(A::LayoutVector, B::SparseArrays.ReadOnly) = equals_layout(A, B)
	==(A::SparseArrays.ReadOnly, B::SubArray{<:Any,1,<:LayoutArray}) = equals_layout(A, B)
	==(A::SubArray{<:Any,1,<:LayoutArray}, B::SparseArrays.ReadOnly) = equals_layout(A, B)
end

end
