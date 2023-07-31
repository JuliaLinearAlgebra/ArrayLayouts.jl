module ArrayLayoutsSparseArraysExt

using SparseArrays

using ArrayLayouts
using ArrayLayouts: dot
import Base: copyto!
import LinearAlgebra

# ambiguity from sparsematrix.jl
copyto!(dest::LayoutMatrix, src::SparseArrays.AbstractSparseMatrixCSC) = _copyto!(dest, src)
copyto!(dest::SubArray{<:Any,2,<:LayoutMatrix}, src::SparseArrays.AbstractSparseMatrixCSC) = _copyto!(dest, src)

@inline LinearAlgebra.dot(a::LayoutArray{<:Number}, b::SparseArrays.SparseVectorUnion{<:Number}) = dot(a,b)
@inline LinearAlgebra.dot(a::SparseArrays.SparseVectorUnion{<:Number}, b::LayoutArray{<:Number}) = dot(a,b)

end # module
