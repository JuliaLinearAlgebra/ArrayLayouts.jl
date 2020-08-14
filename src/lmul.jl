for Typ in (:Lmul, :Rmul)
    @eval begin
        struct $Typ{StyleA, StyleB, TypeA, TypeB}
            A::TypeA
            B::TypeB
        end

        $Typ(A::TypeA, B::TypeB) where {TypeA,TypeB} = $Typ{typeof(MemoryLayout(TypeA)),typeof(MemoryLayout(TypeB)),TypeA,TypeB}(A,B)

        $Typ(M::Mul) = $Typ(M.A, M.B)

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
lmul!(A, B) = materialize!(Lmul(A, B))
rmul!(A, B) = materialize!(Rmul(A, B))

materialize(L::Lmul) = copy(instantiate(L))

copy(M::Lmul) = lmul!(M.A, copyto!(similar(M), M.B))
copy(M::Rmul) = rmul!(copyto!(similar(M), M.A), M.B)

@inline function _lmul_copyto!(dest, M)
    M.B ≡ dest || copyto!(dest, M.B)
    lmul!(M.A,dest)
end

@inline function _rmul_copyto!(dest, M::Rmul)
    M.A ≡ dest || copyto!(dest, M.A)
    rmul!(dest,M.B)
end

copyto!(dest, M::Lmul) = _lmul_copyto!(dest, M)
copyto!(dest::AbstractArray, M::Lmul) = _lmul_copyto!(dest, M)
copyto!(dest, M::Rmul) = _rmul_copyto!(dest, M)
copyto!(dest::AbstractArray, M::Rmul) = _rmul_copyto!(dest, M)

materialize!(M::Lmul) = LinearAlgebra.lmul!(M.A,M.B)
materialize!(M::Rmul) = LinearAlgebra.rmul!(M.A,M.B)

materialize!(M::Lmul{ScalarLayout}) = Base.invoke(LinearAlgebra.lmul!, Tuple{Number,AbstractArray}, M.A, M.B)
materialize!(M::Rmul{<:Any,ScalarLayout}) = Base.invoke(LinearAlgebra.rmul!, Tuple{AbstractArray,Number}, M.A, M.B)

function materialize!(M::Lmul{ScalarLayout,<:SymmetricLayout})
    lmul!(M.A, symmetricdata(M.B))
    M.B
end
function materialize!(M::Lmul{ScalarLayout,<:HermitianLayout})
    lmul!(M.A, hermitiandata(M.B))
    M.B
end
function materialize!(M::Rmul{<:SymmetricLayout,ScalarLayout})
    rmul!(symmetricdata(M.A), M.B)
    M.A
end
function materialize!(M::Rmul{<:HermitianLayout,ScalarLayout})
    rmul!(hermitiandata(M.A), M.B)
    M.A
end

macro _layoutlmul(Typ)
    esc(quote
        LinearAlgebra.lmul!(A::$Typ, x::AbstractVector) = ArrayLayouts.lmul!(A,x)
        LinearAlgebra.lmul!(A::$Typ, x::AbstractMatrix) = ArrayLayouts.lmul!(A,x)
        LinearAlgebra.lmul!(A::$Typ, x::StridedVector) = ArrayLayouts.lmul!(A,x)
        LinearAlgebra.lmul!(A::$Typ, x::StridedMatrix) = ArrayLayouts.lmul!(A,x)
    end)
end

macro layoutlmul(Typ)
    esc(quote
        ArrayLayouts.@_layoutlmul UpperTriangular{T, <:$Typ{T}} where T
        ArrayLayouts.@_layoutlmul UnitUpperTriangular{T, <:$Typ{T}} where T
        ArrayLayouts.@_layoutlmul LowerTriangular{T, <:$Typ{T}} where T
        ArrayLayouts.@_layoutlmul UnitLowerTriangular{T, <:$Typ{T}} where T
    end)
end

macro _layoutrmul(Typ)
    esc(quote
        LinearAlgebra.rmul!(A::AbstractMatrix, B::$Typ) = ArrayLayouts.rmul!(A, B)
        LinearAlgebra.rmul!(A::StridedMatrix, B::$Typ) = ArrayLayouts.rmul!(A, B)
    end)
end

macro layoutrmul(Typ)
    esc(quote
        ArrayLayouts.@_layoutrmul UpperTriangular{T, <:$Typ{T}} where T
        ArrayLayouts.@_layoutrmul UnitUpperTriangular{T, <:$Typ{T}} where T
        ArrayLayouts.@_layoutrmul LowerTriangular{T, <:$Typ{T}} where T
        ArrayLayouts.@_layoutrmul UnitLowerTriangular{T, <:$Typ{T}} where T
    end)
end