struct Mul{StyleA, StyleB, AA, BB}
    A::AA
    B::BB
end

@inline Mul{StyleA,StyleB}(A::AA, B::BB) where {StyleA,StyleB,AA,BB} =
    Mul{StyleA,StyleB,AA,BB}(A,B)

@inline Mul(A, B) = Mul{typeof(MemoryLayout(A)), typeof(MemoryLayout(B))}(A, B)

@inline _mul_eltype(A) = A
@inline _mul_eltype(A, B) = Base.promote_op(*, A, B)
@inline _mul_eltype(A, B, C, D...) = _mul_eltype(Base.promote_op(*, A, B), C, D...)

@inline eltype(M::Mul) = _mul_eltype(eltype(M.A), eltype(M.B))

size(M::Mul, p::Int) = size(M)[p]
axes(M::Mul, p::Int) = axes(M)[p]
length(M::Mul) = prod(size(M))
size(M::Mul) = map(length,axes(M))

_mul_axes(A::Tuple{<:Any,<:Any}, ::Tuple{<:Any}) = (A[1],) # matrix * vector
_mul_axes(A::Tuple{<:Any}, B::Tuple{<:Any,<:Any}) = (A[1],B[2]) # vector * matrix
_mul_axes(A::Tuple{<:Any,<:Any}, B::Tuple{<:Any,<:Any}) = (A[1],B[2]) # matrix * matrix
_mul_axes(::Tuple{}, ::Tuple{}) = () # scalar * scalar
_mul_axes(::Tuple{}, B::Tuple) = B # scalar * B
_mul_axes(A::Tuple, B::Tuple{}) = A # A * scalar

axes(M::Mul) = _mul_axes(axes(M.A), axes(M.B))

# The following design is to support QuasiArrays.jl where indices
# may not be `Int`
function _getindex(::Type{Tuple{AA}}, M::Mul, (k,)::Tuple{AA}) where AA
    A,B = M.A, M.B
    ret = zero(eltype(M))
    for j = rowsupport(A, k) ∩ colsupport(B,1)
        ret += A[k,j] * B[j]
    end
    ret
end

function _getindex(::Type{Tuple{AA,BB}}, M::Mul, (k, j)::Tuple{AA,BB}) where {AA,BB}
    A,B = M.A,M.B
    ret = zero(eltype(M))
    @inbounds for ℓ in (rowsupport(A,k) ∩ colsupport(B,j))
        ret += A[k,ℓ] * B[ℓ,j]
    end
    ret
end

# linear indexing
_getindex(::Type{NTuple{2,Int}}, M, k::Tuple{Int}) = M[Base._ind2sub(axes(M), k...)...]

_getindex(::Type{Tuple{Int}}, M, (k,)::Tuple{CartesianIndex{1}}) = M[convert(Int, k)]
_getindex(::Type{NTuple{2,Int}}, M, (kj,)::Tuple{CartesianIndex{2}}) = M[kj[1], kj[2]]

"""
   indextype(A)

gives the expected index type for an array, or array-like object. For example,
if it is vector-like it will return `Tuple{Int}`, or if it is matrix-like it will
return `Tuple{Int,Int}`. Other types may have non-integer based indexing.
"""
indextype(A) = indextype(axes(A))

indextype(axes::Tuple) = Tuple{map(eltype, axes)...}

getindex(M::Mul, k...) = _getindex(indextype(M), M, k)


"""
   mulreduce(M::Mul)

returns a lower level lazy multiplication object such as `MulAdd`, `Lmul` or `Rmul`.
The Default is `MulAdd`. Note that the lowered type must overload `copyto!` and `copy`.
"""
mulreduce(M::Mul) = MulAdd(M)

similar(M::Mul, ::Type{T}, axes) where {T,N} = similar(mulreduce(M), T, axes)
similar(M::Mul, ::Type{T}) where T = similar(mulreduce(M), T)
similar(M::Mul) = similar(mulreduce(M))

check_mul_axes(A) = nothing
_check_mul_axes(::Number, ::Number) = nothing
_check_mul_axes(::Number, _) = nothing
_check_mul_axes(_, ::Number) = nothing
_check_mul_axes(A, B) = axes(A,2) == axes(B,1) || throw(DimensionMismatch("Second axis of A, $(axes(A,2)), and first axis of B, $(axes(B,1)) must match"))
function check_mul_axes(A, B, C...)
    _check_mul_axes(A, B)
    check_mul_axes(B, C...)
end

# we need to special case AbstractQ as it allows non-compatiple multiplication
function check_mul_axes(A::AbstractQ, B, C...)
    axes(A.factors, 1) == axes(B, 1) || axes(A.factors, 2) == axes(B, 1) ||
        throw(DimensionMismatch("First axis of B, $(axes(B,1)) must match either axes of A, $(axes(A))"))
    check_mul_axes(B, C...)
end

function instantiate(M::Mul)
    @boundscheck check_mul_axes(M.A, M.B)
    M
end

materialize(M::Mul) = copy(instantiate(M))
@inline mul(A, B) = materialize(Mul(A,B))

copy(M::Mul) = copy(mulreduce(M))
@inline copyto!(dest, M::Mul) = copyto!(dest, mulreduce(M))
@inline copyto!(dest::AbstractArray, M::Mul) = copyto!(dest, mulreduce(M))
mul!(dest::AbstractArray, A::AbstractArray, B::AbstractArray) = copyto!(dest, Mul(A,B))
mul!(dest::AbstractArray, A::AbstractArray, B::AbstractArray, α::Number, β::Number) = muladd!(α, A, B, β, dest)


broadcastable(M::Mul) = M


macro veclayoutmul(Typ)
    ret = quote
        Base.:*(A::AbstractMatrix, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::Adjoint{<:Any,<:AbstractMatrix{T}}, B::$Typ{S}) where {T,S} = ArrayLayouts.mul(A,B)
        Base.:*(A::Transpose{<:Any,<:AbstractMatrix{T}}, B::$Typ{S}) where {T,S} = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.AdjointAbsVec, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.TransposeAbsVec, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.AdjointAbsVec{<:Number}, B::$Typ{<:Number}) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.TransposeAbsVec{T}, B::$Typ{T}) where T<:Real = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.TransposeAbsVec{T,<:$Typ{T}}, B::$Typ{T}) where T<:Real = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.TransposeAbsVec{T,<:$Typ{T}}, B::AbstractVector{T}) where T<:Real = ArrayLayouts.mul(A,B)

        Base.:*(A::LinearAlgebra.AbstractQ, B::$Typ) = ArrayLayouts.mul(A,B)
    end
    for Struc in (:AbstractTriangular, :Diagonal)
        ret = quote
            $ret

            Base.:*(A::LinearAlgebra.$Struc, B::$Typ) = ArrayLayouts.mul(A,B)
        end
    end
    for Mod in (:AdjointAbsVec, :TransposeAbsVec)
        ret = quote
            $ret

            LinearAlgebra.mul!(dest::AbstractVector, A::$Mod{<:Any,<:$Typ}, b::AbstractVector) =
                ArrayLayouts.mul!(dest,A,b)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::ArrayLayouts.LayoutMatrix) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractMatrix) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractVector) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Number,<:$Typ}, B::AbstractVector{<:Number}) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Typ) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Number,<:$Typ}, B::$Typ{<:Number}) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::Diagonal) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::LinearAlgebra.AbstractTriangular) = ArrayLayouts.mul(A,B)
        end
    end

    esc(ret)
end

macro layoutmul(Typ)
    ret = quote
        LinearAlgebra.mul!(dest::AbstractVector, A::$Typ, b::AbstractVector) =
            ArrayLayouts.mul!(dest,A,b)

        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, B::AbstractMatrix) =
            ArrayLayouts.mul!(dest,A,B)
        LinearAlgebra.mul!(dest::AbstractMatrix, A::AbstractMatrix, B::$Typ) =
            ArrayLayouts.mul!(dest,A,B)
        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, B::$Typ) =
            ArrayLayouts.mul!(dest,A,B)
        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, B::AbstractMatrix, α::Number, β::Number) =
            ArrayLayouts.mul!(dest,A,B,α,β)
        LinearAlgebra.mul!(dest::AbstractMatrix, A::AbstractMatrix, B::$Typ, α::Number, β::Number) =
            ArrayLayouts.mul!(dest,A,B,α,β)
        LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, B::$Typ, α::Number, β::Number) =
            ArrayLayouts.mul!(dest,A,B,α,β)

        Base.:*(A::$Typ, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::$Typ, B::AbstractMatrix) = ArrayLayouts.mul(A,B)
        Base.:*(A::$Typ, B::AbstractVector) = ArrayLayouts.mul(A,B)
        Base.:*(A::$Typ, B::ArrayLayouts.LayoutVector) = ArrayLayouts.mul(A,B)
        Base.:*(A::AbstractMatrix, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.AdjointAbsVec, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.TransposeAbsVec, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.AdjointAbsVec{<:Any,<:Zeros{<:Any,1}}, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.TransposeAbsVec{<:Any,<:Zeros{<:Any,1}}, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::LinearAlgebra.Transpose{T,<:$Typ}, B::Zeros{T,1}) where T<:Real = ArrayLayouts.mul(A,B)

        Base.:*(A::LinearAlgebra.AbstractQ, B::$Typ) = ArrayLayouts.mul(A,B)
        Base.:*(A::$Typ, B::LinearAlgebra.AbstractQ) = ArrayLayouts.mul(A,B)
    end
    for Struc in (:AbstractTriangular, :Diagonal)
        ret = quote
            $ret

            Base.:*(A::LinearAlgebra.$Struc, B::$Typ) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Typ, B::LinearAlgebra.$Struc) = ArrayLayouts.mul(A,B)
        end
    end
    for Mod in (:Adjoint, :Transpose, :Symmetric, :Hermitian)
        ret = quote
            $ret

            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, b::$Mod{<:Any,<:AbstractMatrix}) =
                ArrayLayouts.mul!(dest,A,b)
            LinearAlgebra.mul!(dest::AbstractVector, A::$Mod{<:Any,<:$Typ}, b::AbstractVector) =
                ArrayLayouts.mul!(dest,A,b)
            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Mod{<:Any,<:$Typ}, B::AbstractMatrix, α::Number, β::Number) =
                ArrayLayouts.mul!(dest,A,B,α,β)
            LinearAlgebra.mul!(dest::AbstractMatrix, A::AbstractMatrix, B::$Mod{<:Any,<:$Typ}, α::Number, β::Number) =
                ArrayLayouts.mul!(dest,A,B,α,β)
            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Mod{<:Any,<:$Typ}, B::$Typ, α::Number, β::Number) =
                ArrayLayouts.mul!(dest,A,B,α,β)
            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, B::$Mod{<:Any,<:$Typ}, α::Number, β::Number) =
                ArrayLayouts.mul!(dest,A,B,α,β)
            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Mod{<:Any,<:AbstractVecOrMat}, B::$Typ, α::Number, β::Number) =
                ArrayLayouts.mul!(dest,A,B,α,β)
            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Typ, B::$Mod{<:Any,<:AbstractVecOrMat}, α::Number, β::Number) =
                ArrayLayouts.mul!(dest,A,B,α,β)
            LinearAlgebra.mul!(dest::AbstractMatrix, A::$Mod{<:Any,<:$Typ}, B::$Mod{<:Any,<:$Typ}, α::Number, β::Number) =
                ArrayLayouts.mul!(dest,A,B,α,β)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractMatrix) = ArrayLayouts.mul(A,B)
            Base.:*(A::AbstractMatrix, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)
            Base.:*(A::LinearAlgebra.AdjointAbsVec, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)
            Base.:*(A::LinearAlgebra.TransposeAbsVec, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::AbstractVector) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::ArrayLayouts.LayoutVector) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::Zeros{<:Any,1}) = ArrayLayouts.mul(A,B)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::$Typ) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Typ, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)

            Base.:*(A::$Mod{<:Any,<:$Typ}, B::Diagonal) = ArrayLayouts.mul(A,B)
            Base.:*(A::Diagonal, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)

            Base.:*(A::LinearAlgebra.AbstractTriangular, B::$Mod{<:Any,<:$Typ}) = ArrayLayouts.mul(A,B)
            Base.:*(A::$Mod{<:Any,<:$Typ}, B::LinearAlgebra.AbstractTriangular) = ArrayLayouts.mul(A,B)
        end
    end

    esc(ret)
end

@veclayoutmul LayoutVector


###
# Dot
###
"""
    Dot(A, B)

is a lazy version of `dot(A, B)`, designed to support
materializing based on `MemoryLayout`.
"""
struct Dot{StyleA,StyleB,ATyp,BTyp}
    A::ATyp
    B::BTyp
end

@inline Dot(A::ATyp,B::BTyp) where {ATyp,BTyp} = Dot{typeof(MemoryLayout(ATyp)), typeof(MemoryLayout(BTyp)), ATyp, BTyp}(A, B)
@inline copy(d::Dot) = Base.invoke(LinearAlgebra.dot, Tuple{AbstractArray,AbstractArray}, d.A, d.B)
@inline materialize(d::Dot) = copy(instantiate(d))
@inline Dot(M::Mul{<:DualLayout,<:Any,<:AbstractMatrix,<:AbstractVector}) = Dot(M.A', M.B)
@inline mulreduce(M::Mul{<:DualLayout,<:Any,<:AbstractMatrix,<:AbstractVector}) = Dot(M)
@inline eltype(D::Dot) = promote_type(eltype(D.A), eltype(D.B))

dot(a, b) = materialize(Dot(a, b))
@inline LinearAlgebra.dot(a::LayoutArray, b::LayoutArray) = dot(a,b)
@inline LinearAlgebra.dot(a::LayoutArray, b::AbstractArray) = dot(a,b)
@inline LinearAlgebra.dot(a::AbstractArray, b::LayoutArray) = dot(a,b)
@inline LinearAlgebra.dot(a::LayoutVector, b::AbstractFill{<:Any,1}) = FillArrays._fill_dot(a,b)
@inline LinearAlgebra.dot(a::AbstractFill{<:Any,1}, b::LayoutVector) = FillArrays._fill_dot(a,b)
@inline LinearAlgebra.dot(a::LayoutArray{<:Number}, b::SparseArrays.SparseVectorUnion{<:Number}) = dot(a,b)
@inline LinearAlgebra.dot(a::SparseArrays.SparseVectorUnion{<:Number}, b::LayoutArray{<:Number}) = dot(a,b)

@inline LinearAlgebra.dot(a::SubArray{<:Any,N,<:LayoutArray}, b::AbstractArray) where N = dot(a,b)
@inline LinearAlgebra.dot(a::SubArray{<:Any,N,<:LayoutArray}, b::LayoutArray) where N = dot(a,b)
@inline LinearAlgebra.dot(a::AbstractArray, b::SubArray{<:Any,N,<:LayoutArray}) where N = dot(a,b)
@inline LinearAlgebra.dot(a::LayoutArray, b::SubArray{<:Any,N,<:LayoutArray}) where N = dot(a,b)
@inline LinearAlgebra.dot(a::SubArray{<:Any,N,<:LayoutArray}, b::SubArray{<:Any,N,<:LayoutArray}) where N = dot(a,b)

# Temporary until layout 3-arg dot is added.
# We go to generic fallback as layout-arrays are structured
dot(x, A, y) = dot(x, mul(A, y))
LinearAlgebra.dot(x::AbstractVector, A::LayoutMatrix, y::AbstractVector) = dot(x, A, y)
LinearAlgebra.dot(x::AbstractVector, A::Symmetric{<:Real,<:LayoutMatrix}, y::AbstractVector) = dot(x, A, y)



# allow overloading for infinite or lazy case
@inline _power_by_squaring(_, _, A, p) = Base.invoke(Base.power_by_squaring, Tuple{AbstractMatrix,Integer}, A, p)
# TODO: Remove unnecessary _apply
_apply(_, _, op, Λ::UniformScaling, A::AbstractMatrix) = op(Diagonal(Fill(Λ.λ,size(A,1))), A)
_apply(_, _, op, A::AbstractMatrix, Λ::UniformScaling) = op(A, Diagonal(Fill(Λ.λ,size(A,1))))

for Typ in (:LayoutMatrix, :(Symmetric{<:Any,<:LayoutMatrix}), :(Hermitian{<:Any,<:LayoutMatrix}),
            :(Adjoint{<:Any,<:LayoutMatrix}), :(Transpose{<:Any,<:LayoutMatrix}))
    @eval begin
        @inline Base.power_by_squaring(A::$Typ, p::Integer) = _power_by_squaring(MemoryLayout(A), size(A), A, p)
        @inline +(A::$Typ, Λ::UniformScaling) = _apply(MemoryLayout(A), size(A), +, A, Λ)
        @inline +(Λ::UniformScaling, A::$Typ) = _apply(MemoryLayout(A), size(A), +, Λ, A)
        @inline -(A::$Typ, Λ::UniformScaling) = _apply(MemoryLayout(A), size(A), -, A, Λ)
        @inline -(Λ::UniformScaling, A::$Typ) = _apply(MemoryLayout(A), size(A), -, Λ, A)
    end
end