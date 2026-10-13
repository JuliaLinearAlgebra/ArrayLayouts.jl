module ArrayLayoutsSparseArraysExt

using ArrayLayouts
using ArrayLayouts: copyto!_layout, equals_layout, sparse_layout, LayoutVecOrMat, UpperOrLowerTriangular
using SparseArrays
# Specifying the full namespace is necessary because of https://github.com/JuliaLang/julia/issues/48533
# See https://github.com/JuliaStats/LogExpFunctions.jl/pull/63
import ArrayLayouts.LinearAlgebra
using ArrayLayouts.LinearAlgebra: Adjoint, Transpose, Symmetric, Hermitian, Diagonal, UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular,
								AdjOrTrans, HermOrSym

import Base: copyto!, ==

# use colsupport to deduce the sparsity pattern
ArrayLayouts.sparse_layout(_, A) = sparse_colsupport(A)

function sparse_colsupport(A::AbstractMatrix{T}) where T
	Base.require_one_based_indexing(A)
	m, n = size(A)
	colptr = Vector{Int}(undef, n+1)
	rowval = Int[]
	nzval = T[]
	colptr[1] = 1
	for j = 1:n
		for k in colsupport(A, j)
			v = A[k, j]
			if !iszero(v)
				push!(rowval, k)
				push!(nzval, v)
			end
		end
		colptr[j+1] = length(rowval) + 1
	end
	SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function sparse_colsupport(a::AbstractVector{T}) where T
	Base.require_one_based_indexing(a)
	nzind = Int[]
	nzval = T[]
	for k in colsupport(a, 1)
		v = a[k]
		if !iszero(v)
			push!(nzind, k)
			push!(nzval, v)
		end
	end
	SparseVector(length(a), nzind, nzval)
end

const LayoutSparseVectors = Union{LayoutVector, SubArray{<:Any,1,<:LayoutArray}}
const LayoutSparseMatrixParents = Union{LayoutMatrix, SubArray{<:Any,2,<:LayoutArray}, AdjOrTrans{<:Any,<:LayoutVecOrMat}}
const LayoutSparseMatrices = Union{LayoutSparseMatrixParents,
									HermOrSym{<:Any,<:LayoutMatrix},
									UpperOrLowerTriangular{<:Any,<:LayoutSparseMatrixParents},
									Diagonal{<:Any,<:LayoutVector}}

SparseArrays.sparse(a::LayoutSparseVectors) = sparse_layout(a)
SparseArrays.sparse(A::LayoutSparseMatrices) = sparse_layout(A)

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
