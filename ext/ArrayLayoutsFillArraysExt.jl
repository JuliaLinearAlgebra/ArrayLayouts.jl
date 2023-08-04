module ArrayLayoutsFillArraysExt

using FillArrays
using FillArrays: AbstractFill, getindex_value

using ArrayLayouts
using ArrayLayouts: OnesLayout, Mul, MulAdd
import ArrayLayouts: MemoryLayout, _copyto!, sub_materialize, diagonaldata
export layoutfillmul, mulzeros

import Base: copy, *
import Base.Broadcast: materialize!
import LinearAlgebra
using LinearAlgebra: Adjoint, Transpose, Symmetric, Hermitian, Diagonal,
    AdjointAbsVec, TransposeAbsVec, UniformScaling

macro layoutfillmul(Typ)
    ret = quote
        (*)(A::LinearAlgebra.AdjointAbsVec{<:Any,<:Zeros{<:Any,1}}, B::$Typ) = ArrayLayouts.mul(A,B)
        (*)(A::LinearAlgebra.TransposeAbsVec{<:Any,<:Zeros{<:Any,1}}, B::$Typ) = ArrayLayouts.mul(A,B)
        (*)(A::LinearAlgebra.Transpose{T,<:$Typ}, B::Zeros{T,1}) where T<:Real = ArrayLayouts.mul(A,B)
    end
    for Mod in (:Adjoint, :Transpose, :Symmetric, :Hermitian)
        ret = quote
            $ret

            (*)(A::$Mod{<:Any,<:$Typ}, B::Zeros{<:Any,1}) = ArrayLayouts.mul(A,B)
        end
    end
    esc(ret)
end

@layoutfillmul LayoutMatrix

*(a::Zeros{<:Any,2}, b::LayoutMatrix) = FillArrays.mult_zeros(a, b)
*(a::LayoutMatrix, b::Zeros{<:Any,2}) = FillArrays.mult_zeros(a, b)
*(a::LayoutMatrix, b::Zeros{<:Any,1}) = FillArrays.mult_zeros(a, b)
*(a::Transpose{T, <:LayoutMatrix{T}} where T, b::Zeros{<:Any, 2}) = FillArrays.mult_zeros(a, b)
*(a::Adjoint{T, <:LayoutMatrix{T}} where T, b::Zeros{<:Any, 2}) = FillArrays.mult_zeros(a, b)
*(A::Adjoint{<:Any, <:Zeros{<:Any,1}}, B::Diagonal{<:Any,<:LayoutVector}) = (B' * A')'
*(A::Transpose{<:Any, <:Zeros{<:Any,1}}, B::Diagonal{<:Any,<:LayoutVector}) = transpose(transpose(B) * transpose(A))

# equivalent to rescaling
function materialize!(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout}})
    M.B .= getindex_value(M.A.diag) .* M.B
    M.B
end
# equivalent to rescaling
function materialize!(M::Rmul{<:Any,<:DiagonalLayout{<:AbstractFillLayout}})
    M.A .= M.A .* getindex_value(M.B.diag)
    M.A
end

copy(M::Ldiv{<:DiagonalLayout{<:AbstractFillLayout}}) = inv(getindex_value(M.A.diag)) .* M.B
copy(M::Ldiv{<:DiagonalLayout{<:AbstractFillLayout},<:DiagonalLayout}) = diagonal(inv(getindex_value(M.A.diag)) .* M.B.diag)

copy(M::Rdiv{<:Any,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A .* inv(getindex_value(M.B.diag))
copy(M::Rdiv{<:DiagonalLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = diagonal(M.A.diag .* inv(getindex_value(M.B.diag)))

copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout}}) = getindex_value(diagonaldata(M.A)) * M.B
copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout},<:DiagonalLayout}) = getindex_value(diagonaldata(M.A)) * M.B
copy(M::Rmul{<:Any,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A * getindex_value(diagonaldata(M.B))
copy(M::Rmul{<:DualLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A * getindex_value(diagonaldata(M.B))

copy(M::Rmul{<:BidiagonalLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A * getindex_value(diagonaldata(M.B))
copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout},<:BidiagonalLayout}) = getindex_value(diagonaldata(M.A)) * M.B
copy(M::Rmul{<:TridiagonalLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A * getindex_value(diagonaldata(M.B))
copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout},<:TridiagonalLayout}) = getindex_value(diagonaldata(M.A)) * M.B
copy(M::Rmul{<:SymTridiagonalLayout,<:DiagonalLayout{<:AbstractFillLayout}}) = M.A * getindex_value(diagonaldata(M.B))
copy(M::Lmul{<:DiagonalLayout{<:AbstractFillLayout},<:SymTridiagonalLayout}) = getindex_value(diagonaldata(M.A)) * M.B

MemoryLayout(::Type{<:AbstractFill}) = FillLayout()
MemoryLayout(::Type{<:Zeros}) = ZerosLayout()
MemoryLayout(::Type{<:Ones}) = OnesLayout()

_copyto!(_, ::AbstractFillLayout, dest::AbstractArray{<:Any,N}, src::AbstractArray{<:Any,N}) where N =
    fill!(dest, getindex_value(src))

    _fill_copyto!(dest, C::Zeros) = zero!(dest) # exploit special fill! overload

sub_materialize(::AbstractFillLayout, V, ax) = Fill(getindex_value(V), ax)
sub_materialize(::ZerosLayout, V, ax) = Zeros{eltype(V)}(ax)
sub_materialize(::OnesLayout, V, ax) = Ones{eltype(V)}(ax)

*(x::AdjointAbsVec{<:Any,<:Zeros{<:Any,1}},   D::Diagonal, y::LayoutVector) = FillArrays._triple_zeromul(x, D, y)
*(x::TransposeAbsVec{<:Any,<:Zeros{<:Any,1}}, D::Diagonal, y::LayoutVector) = FillArrays._triple_zeromul(x, D, y)

@inline LinearAlgebra.dot(a::LayoutVector, b::AbstractFill{<:Any,1}) = FillArrays._fill_dot_rev(a,b)
@inline LinearAlgebra.dot(a::AbstractFill{<:Any,1}, b::LayoutVector) = FillArrays._fill_dot(a,b)

# TODO: Remove unnecessary _apply
_apply(_, _, op, Λ::UniformScaling, A::AbstractMatrix) = op(Diagonal(Fill(Λ.λ,(axes(A,1),))), A)
_apply(_, _, op, A::AbstractMatrix, Λ::UniformScaling) = op(A, Diagonal(Fill(Λ.λ,(axes(A,1),))))

# equivalent to rescaling
function materialize!(M::MulAdd{<:DiagonalLayout{<:AbstractFillLayout}})
    checkdimensions(M)
    M.C .= (M.α * getindex_value(M.A.diag)) .* M.B .+ M.β .* M.C
    M.C
end

function materialize!(M::MulAdd{<:Any,<:DiagonalLayout{<:AbstractFillLayout}})
    checkdimensions(M)
    M.C .= M.α .* M.A .* getindex_value(M.B.diag) .+ M.β .* M.C
    M.C
end

fillzeros(::Type{T}, ax) where T<:Number = Zeros{T}(ax)
mulzeros(::Type{T}, M) where T<:Number = fillzeros(T, axes(M))
mulzeros(::Type{T}, M::Mul{<:DualLayout,<:Any,<:Adjoint}) where T<:Number = fillzeros(T, axes(M,2))'
mulzeros(::Type{T}, M::Mul{<:DualLayout,<:Any,<:Transpose}) where T<:Number = transpose(fillzeros(T, axes(M,2)))

end
