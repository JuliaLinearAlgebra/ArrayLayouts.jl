for Typ in (:Lmul, :Rmul)
    @eval begin
        struct $Typ{StyleA, StyleB, TypeA, TypeB}
            A::TypeA
            B::TypeB
        end

        $Typ(A::TypeA, B::TypeB) where {TypeA,TypeB} = $Typ{typeof(MemoryLayout(TypeA)),typeof(MemoryLayout(TypeB)),TypeA,TypeB}(A,B)

        BroadcastStyle(::Type{<:$Typ}) = ApplyBroadcastStyle()
        broadcastable(M::$Typ) = M

        eltype(::$Typ{<:Any,<:Any,A,B}) where {A,B} = promote_type(eltype(A), eltype(B))
        size(M::$Typ, p::Int) = size(M)[p]
        axes(M::$Typ, p::Int) = axes(M)[p]
        length(M::$Typ) = prod(size(M))
        size(M::$Typ) = map(length,axes(M))
        axes(M::$Typ) = (axes(M.A,1),axes(M.B,2))

        similar(M::$Typ, ::Type{T}, axes) where {T,N} = similar(Array{T}, axes)
        similar(M::$Typ, ::Type{T}) where T = similar(M, T, axes(M))
        similar(M::$Typ) = similar(M, eltype(M))
    end
end



const MatLmulVec{StyleA,StyleB} = Lmul{StyleA,StyleB,<:AbstractMatrix,<:AbstractVector}
const MatLmulMat{StyleA,StyleB} = Lmul{StyleA,StyleB,<:AbstractMatrix,<:AbstractMatrix}

const BlasMatLmulVec{StyleA,StyleB,T<:BlasFloat} = Lmul{StyleA,StyleB,<:AbstractMatrix{T},<:AbstractVector{T}}
const BlasMatLmulMat{StyleA,StyleB,T<:BlasFloat} = Lmul{StyleA,StyleB,<:AbstractMatrix{T},<:AbstractMatrix{T}}

const MatRmulMat{StyleA,StyleB} = Rmul{StyleA,StyleB,<:AbstractMatrix,<:AbstractMatrix}
const BlasMatRmulMat{StyleA,StyleB,T<:BlasFloat} = Rmul{StyleA,StyleB,<:AbstractMatrix{T},<:AbstractMatrix{T}}


####
# LMul materialize
####

axes(M::MatLmulVec) = (axes(M.A,1),)

lmul(A, B) = materialize(Lmul(A, B))

materialize(L::Lmul) = copy(instantiate(L))

copy(M::Lmul) = materialize!(Lmul(M.A,copy(M.B)))
copy(M::Rmul) = materialize!(Rmul(copy(M.A),M.B))

@inline function copyto!(dest::AbstractArray, M::Lmul)
    M.B ≡ dest || copyto!(dest, M.B)
    materialize!(Lmul(M.A,dest))
end

@inline function copyto!(dest::AbstractArray, M::Rmul)
    M.A ≡ dest || copyto!(dest, M.A)
    materialize!(Rmul(dest,M.B))
end

materialize!(M::Lmul) = lmul!(M.A,M.B)
materialize!(M::Rmul) = rmul!(M.A,M.B)



macro lazylmul(Typ)
    esc(quote
        LinearAlgebra.lmul!(A::$Typ, x::AbstractVector) = ArrayLayouts.materialize!(ArrayLayouts.Lmul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::AbstractMatrix) = ArrayLayouts.materialize!(ArrayLayouts.Lmul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::StridedVector) = ArrayLayouts.materialize!(ArrayLayouts.Lmul(A,x))
        LinearAlgebra.lmul!(A::$Typ, x::StridedMatrix) = ArrayLayouts.materialize!(ArrayLayouts.Lmul(A,x))
    end)
end



