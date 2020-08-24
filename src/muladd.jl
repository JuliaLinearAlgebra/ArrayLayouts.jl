### This support BLAS style multiplication
#           α * A * B + β C
# but avoids the broadcast machinery

# Lazy representation of α*A*B + β*C
struct MulAdd{StyleA, StyleB, StyleC, T, AA, BB, CC}
    α::T
    A::AA
    B::BB
    β::T
    C::CC
end

@inline MulAdd{StyleA,StyleB,StyleC}(α::T, A::AA, B::BB, β::T, C::CC) where {StyleA,StyleB,StyleC,T,AA,BB,CC} =
    MulAdd{StyleA,StyleB,StyleC,T,AA,BB,CC}(α,A,B,β,C)

@inline function MulAdd{StyleA,StyleB,StyleC}(αT, A, B, βV, C) where {StyleA,StyleB,StyleC}
    α,β = promote(αT,βV)
    MulAdd{StyleA,StyleB,StyleC}(α, A, B, β, C)
end

@inline MulAdd(α, A::AA, B::BB, β, C::CC) where {AA,BB,CC} =
    MulAdd{typeof(MemoryLayout(AA)), typeof(MemoryLayout(BB)), typeof(MemoryLayout(CC))}(α, A, B, β, C)

MulAdd(A, B) = MulAdd(Mul(A, B))
function MulAdd(M::Mul)
    TV = eltype(M)
    MulAdd(scalarone(TV), M.A, M.B, scalarzero(TV), mulzeros(TV,M))
end

@inline eltype(::MulAdd{StyleA,StyleB,StyleC,T,AA,BB,CC}) where {StyleA,StyleB,StyleC,T,AA,BB,CC} =
     promote_type(_mul_eltype(T, eltype(AA), eltype(BB)), _mul_eltype(T, eltype(CC)))


size(M::MulAdd, p::Int) = size(M)[p]
axes(M::MulAdd, p::Int) = axes(M)[p]
length(M::MulAdd) = prod(size(M))
size(M::MulAdd) = map(length,axes(M))
axes(M::MulAdd) = axes(M.C)

similar(M::MulAdd, ::Type{T}, axes) where {T,N} = similar(Array{T}, axes)
similar(M::MulAdd, ::Type{T}) where T = similar(M, T, axes(M))
similar(M::MulAdd) = similar(M, eltype(M))

function checkdimensions(M::MulAdd)
    @boundscheck check_mul_axes(M.α, M.A, M.B)
    @boundscheck check_mul_axes(M.β, M.C)
    @boundscheck axes(M.A,1) == axes(M.C,1) || throw(DimensionMismatch("First axis of A, $(axes(M.A,1)), and first axis of C, $(axes(M.C,1)) must match"))
    @boundscheck axes(M.B,2) == axes(M.C,2) || throw(DimensionMismatch("Second axis of B, $(axes(M.B,2)), and second axis of C, $(axes(M.C,2)) must match"))
end
@propagate_inbounds function instantiate(M::MulAdd)
    checkdimensions(M)
    M
end

const ArrayMulArrayAdd{StyleA,StyleB,StyleC} = MulAdd{StyleA,StyleB,StyleC,<:Any,<:AbstractArray,<:AbstractArray,<:AbstractArray}
const MatMulVecAdd{StyleA,StyleB,StyleC} = MulAdd{StyleA,StyleB,StyleC,<:Any,<:AbstractMatrix,<:AbstractVector,<:AbstractVector}
const MatMulMatAdd{StyleA,StyleB,StyleC} = MulAdd{StyleA,StyleB,StyleC,<:Any,<:AbstractMatrix,<:AbstractMatrix,<:AbstractMatrix}
const VecMulMatAdd{StyleA,StyleB,StyleC} = MulAdd{StyleA,StyleB,StyleC,<:Any,<:AbstractVector,<:AbstractMatrix,<:AbstractMatrix}

broadcastable(M::MulAdd) = M


const BlasMatMulVecAdd{StyleA,StyleB,StyleC,T<:BlasFloat} = MulAdd{StyleA,StyleB,StyleC,T,<:AbstractMatrix{T},<:AbstractVector{T},<:AbstractVector{T}}
const BlasMatMulMatAdd{StyleA,StyleB,StyleC,T<:BlasFloat} = MulAdd{StyleA,StyleB,StyleC,T,<:AbstractMatrix{T},<:AbstractMatrix{T},<:AbstractMatrix{T}}
const BlasVecMulMatAdd{StyleA,StyleB,StyleC,T<:BlasFloat} = MulAdd{StyleA,StyleB,StyleC,T,<:AbstractVector{T},<:AbstractMatrix{T},<:AbstractMatrix{T}}

muladd!(α, A, B, β, C) = materialize!(MulAdd(α, A, B, β, C))
materialize(M::MulAdd) = copy(instantiate(M))
copy(M::MulAdd) = copyto!(similar(M), M)

_fill_copyto!(dest, C) = copyto!(dest, C)
_fill_copyto!(dest, C::Zeros) = zero!(dest) # exploit special fill! overload

@inline copyto!(dest::AbstractArray{T}, M::MulAdd) where T = 
    muladd!(M.α, unalias(dest,M.A), unalias(dest,M.B), M.β, _fill_copyto!(dest, M.C))

# Modified from LinearAlgebra._generic_matmatmul!
function tile_size(T, S, R)
    tile_size = 0
    if isbitstype(R) && isbitstype(T) && isbitstype(S)
        tile_size = floor(Int, sqrt(tilebufsize / max(sizeof(R), sizeof(S), sizeof(T))))
    end
    tile_size
end

function tiled_blasmul!(tile_size, α, A::AbstractMatrix{T}, B::AbstractMatrix{S}, β, C::AbstractMatrix{R}) where {S,T,R}
    mA, nA = size(A)
    mB, nB = size(B)
    nA == mB || throw(DimensionMismatch("Dimensions must match"))
    size(C) == (mA, nB) || throw(DimensionMismatch("Dimensions must match"))

    (iszero(mA) || iszero(nB)) && return C
    iszero(nA) && return lmul!(β, C)

    @inbounds begin
        sz = (tile_size, tile_size)
        # FIXME: This code is completely invalid!!!
        Atile = unsafe_wrap(Array, convert(Ptr{T}, pointer(Abuf[Threads.threadid()])), sz)
        Btile = unsafe_wrap(Array, convert(Ptr{S}, pointer(Bbuf[Threads.threadid()])), sz)

        z1 = zero(A[1, 1]*B[1, 1] + A[1, 1]*B[1, 1])
        z = convert(promote_type(typeof(z1), R), z1)

        if mA < tile_size && nA < tile_size && nB < tile_size
            copy_transpose!(Atile, 1:nA, 1:mA, 'N', A, 1:mA, 1:nA)
            copyto!(Btile, 1:mB, 1:nB, 'N', B, 1:mB, 1:nB)
            for j = 1:nB
                boff = (j-1)*tile_size
                for i = 1:mA
                    aoff = (i-1)*tile_size
                    s = z
                    for k = 1:nA
                        s += Atile[aoff+k] * Btile[boff+k]
                    end
                    C[i,j] = α*s + β*C[i,j]
                end
            end
        else
            # FIXME: This code is completely invalid!!!
            Ctile = unsafe_wrap(Array, convert(Ptr{R}, pointer(Cbuf[Threads.threadid()])), sz)
            for jb = 1:tile_size:nB
                jlim = min(jb+tile_size-1,nB)
                jlen = jlim-jb+1
                for ib = 1:tile_size:mA
                    ilim = min(ib+tile_size-1,mA)
                    ilen = ilim-ib+1
                    copyto!(Ctile, 1:ilen, 1:jlen, C, ib:ilim, jb:jlim)
                    lmul!(β,Ctile)
                    for kb = 1:tile_size:nA
                        klim = min(kb+tile_size-1,mB)
                        klen = klim-kb+1
                        copy_transpose!(Atile, 1:klen, 1:ilen, 'N', A, ib:ilim, kb:klim)
                        copyto!(Btile, 1:klen, 1:jlen, 'N', B, kb:klim, jb:jlim)
                        for j=1:jlen
                            bcoff = (j-1)*tile_size
                            for i = 1:ilen
                                aoff = (i-1)*tile_size
                                s = z
                                for k = 1:klen
                                    s += Atile[aoff+k] * Btile[bcoff+k]
                                end
                                Ctile[bcoff+i] += α*s
                            end
                        end
                    end
                    copyto!(C, ib:ilim, jb:jlim, Ctile, 1:ilen, 1:jlen)
                end
            end
        end
    end

    C
end

function default_blasmul!(α, A::AbstractMatrix, B::AbstractMatrix, β, C::AbstractMatrix)
    mA, nA = size(A)
    mB, nB = size(B)
    nA == mB || throw(DimensionMismatch("Dimensions must match"))
    size(C) == (mA, nB) || throw(DimensionMismatch("Dimensions must match"))

    lmul!(β, C)

    (iszero(mA) || iszero(nB)) && return C
    iszero(nA) && return C

    @inbounds for k in colsupport(A), j in rowsupport(B,rowsupport(A,k))
        z2 = zero(A[k, 1]*B[1, j] + A[k, 1]*B[1, j])
        Ctmp = convert(promote_type(eltype(C), typeof(z2)), z2)
        @simd for ν = rowsupport(A,k) ∩ colsupport(B,j)
            Ctmp = muladd(A[k, ν],B[ν, j],Ctmp)
        end
        C[k,j] = muladd(α,Ctmp, C[k,j])
    end
    C
end

function _default_blasmul!(::IndexLinear, α, A::AbstractMatrix, B::AbstractVector, β, C::AbstractVector)
    mA, nA = size(A)
    mB = length(B)
    nA == mB || throw(DimensionMismatch("Dimensions must match"))
    length(C) == mA || throw(DimensionMismatch("Dimensions must match"))

    lmul!(β, C)
    (nA == 0 || mB == 0)  && return C

    z = zero(A[1]*B[1] + A[1]*B[1])
    Astride = size(A, 1) # use size, not stride, since its not pointer arithmetic

    @inbounds for k in colsupport(B,1)
        aoffs = (k-1)*Astride
        b = B[k]
        for i = 1:mA
            C[i] += α * A[aoffs + i] * b
        end
    end

    C
end

function _default_blasmul!(::IndexCartesian, α, A::AbstractMatrix, B::AbstractVector, β, C::AbstractVector)
    mA, nA = size(A)
    mB = length(B)
    nA == mB || throw(DimensionMismatch("Dimensions must match"))
    length(C) == mA || throw(DimensionMismatch("Dimensions must match"))

    lmul!(β, C)
    (nA == 0 || mB == 0)  && return C

    z = zero(A[1,1]*B[1] + A[1,1]*B[1])

    @inbounds for k in colsupport(B,1)
        b = B[k]
        for i = colsupport(A,k)
            C[i] += α * A[i,k] * b
        end
    end

    C
end

default_blasmul!(α, A::AbstractMatrix, B::AbstractVector, β, C::AbstractVector) =
    _default_blasmul!(Base.IndexStyle(typeof(A)), α, A, B, β, C)

function materialize!(M::MatMulMatAdd)
    α, A, B, β, C = M.α, M.A, M.B, M.β, M.C
    default_blasmul!(α, unalias(C,A), unalias(C,B), iszero(β) ? false : β, C)
end

function materialize!(M::MatMulMatAdd{<:AbstractStridedLayout,<:AbstractStridedLayout,<:AbstractStridedLayout})
    α, Ain, Bin, β, C = M.α, M.A, M.B, M.β, M.C
    A = unalias(C, Ain)
    B = unalias(C, Bin)
    ts = tile_size(eltype(A), eltype(B), eltype(C))
    if iszero(β) # false is a "strong" zero to wipe out NaNs
        if ts == 0 || !(axes(A) isa NTuple{2,OneTo{Int}}) || !(axes(B) isa NTuple{2,OneTo{Int}}) || !(axes(C) isa NTuple{2,OneTo{Int}})
            default_blasmul!(α, A, B, false, C)
        else
            tiled_blasmul!(ts, α, A, B, false, C)
        end
    else
        if ts == 0 || !(axes(A) isa NTuple{2,OneTo{Int}}) || !(axes(B) isa NTuple{2,OneTo{Int}}) || !(axes(C) isa NTuple{2,OneTo{Int}})
            default_blasmul!(α, A, B, β, C)
        else
            tiled_blasmul!(ts, α, A, B, β, C)
        end
    end
end

function materialize!(M::MatMulVecAdd)
    α, A, B, β, C = M.α, M.A, M.B, M.β, M.C
    default_blasmul!(α, unalias(C,A), unalias(C,B), iszero(β) ? false : β, C)
end

@inline _gemv!(tA, α, A, x, β, y) = BLAS.gemv!(tA, α, unalias(y,A), unalias(y,x), β, y)
@inline _gemm!(tA, tB, α, A, B, β, C) = BLAS.gemm!(tA, tB, α, unalias(C,A), unalias(C,B), β, C)

# work around pointer issues
@inline materialize!(M::BlasMatMulVecAdd{<:AbstractColumnMajor,<:AbstractStridedLayout,<:AbstractStridedLayout}) =
    _gemv!('N', M.α, M.A, M.B, M.β, M.C)
@inline materialize!(M::BlasMatMulVecAdd{<:AbstractRowMajor,<:AbstractStridedLayout,<:AbstractStridedLayout}) =
    _gemv!('T', M.α, transpose(M.A), M.B, M.β, M.C)
@inline materialize!(M::BlasMatMulVecAdd{<:ConjLayout{<:AbstractRowMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasComplex}) =
    _gemv!('C', M.α, adjoint(M.A), M.B, M.β, M.C)

@inline materialize!(M::BlasVecMulMatAdd{<:AbstractColumnMajor,<:AbstractColumnMajor,<:AbstractColumnMajor}) =
    _gemm!('N', 'N', M.α, M.A, M.B, M.β, M.C)
@inline materialize!(M::BlasVecMulMatAdd{<:AbstractColumnMajor,<:AbstractRowMajor,<:AbstractColumnMajor}) =
    _gemm!('N', 'T', M.α, M.A, transpose(M.B), M.β, M.C)
@inline materialize!(M::BlasVecMulMatAdd{<:AbstractColumnMajor,<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('N', 'C', M.α, M.A, adjoint(M.B), M.β, M.C)

@inline materialize!(M::BlasMatMulMatAdd{<:AbstractColumnMajor,<:AbstractColumnMajor,<:AbstractColumnMajor}) =
    _gemm!('N', 'N', M.α, M.A, M.B, M.β, M.C)
@inline materialize!(M::BlasMatMulMatAdd{<:AbstractColumnMajor,<:AbstractRowMajor,<:AbstractColumnMajor}) =
    _gemm!('N', 'T', M.α, M.A, transpose(M.B), M.β, M.C)
@inline materialize!(M::BlasMatMulMatAdd{<:AbstractColumnMajor,<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('N', 'C', M.α, M.A, adjoint(M.B), M.β, M.C)

@inline materialize!(M::BlasMatMulMatAdd{<:AbstractRowMajor,<:AbstractColumnMajor,<:AbstractColumnMajor}) =
    _gemm!('T', 'N', M.α, transpose(M.A), M.B, M.β, M.C)
@inline materialize!(M::BlasMatMulMatAdd{<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('C', 'N', M.α, adjoint(M.A), M.B, M.β, M.C)

@inline materialize!(M::BlasMatMulMatAdd{<:AbstractRowMajor,<:AbstractRowMajor,<:AbstractColumnMajor}) =
    _gemm!('T', 'T', M.α, transpose(M.A), transpose(M.B), M.β, M.C)
@inline materialize!(M::BlasMatMulMatAdd{<:AbstractRowMajor,<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('T', 'C', M.α, transpose(M.A), adjoint(M.B), M.β, M.C)

@inline materialize!(M::BlasMatMulMatAdd{<:ConjLayout{<:AbstractRowMajor},<:AbstractRowMajor,<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('C', 'T', M.α, adjoint(M.A), transpose(M.B), M.β, M.C)
@inline materialize!(M::BlasMatMulMatAdd{<:ConjLayout{<:AbstractRowMajor},<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:BlasComplex}) =
    _gemm!('C', 'C', M.α, adjoint(M.A), adjoint(M.B), M.β, M.C)

@inline materialize!(M::BlasMatMulMatAdd{<:AbstractColumnMajor,<:AbstractColumnMajor,<:AbstractRowMajor}) =
    _gemm!('T', 'T', M.α, M.B, M.A, M.β, transpose(M.C))
@inline materialize!(M::BlasMatMulMatAdd{<:AbstractColumnMajor,<:AbstractColumnMajor,<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('C', 'C', M.α, M.B, M.A, M.β, adjoint(M.C))

@inline materialize!(M::BlasMatMulMatAdd{<:AbstractColumnMajor,<:AbstractRowMajor,<:AbstractRowMajor}) =
    _gemm!('N', 'T', M.α, transpose(M.B), M.A, M.β, transpose(M.C))
@inline materialize!(M::BlasMatMulMatAdd{<:AbstractColumnMajor,<:AbstractRowMajor,<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('N', 'T', M.α, transpose(M.B), M.A, M.β, M.C')
@inline materialize!(M::BlasMatMulMatAdd{<:AbstractColumnMajor,<:ConjLayout{<:AbstractRowMajor},<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('N', 'C', M.α, adjoint(M.B), M.A, M.β, adjoint(M.C))

@inline materialize!(M::BlasMatMulMatAdd{<:AbstractRowMajor,<:AbstractColumnMajor,<:AbstractRowMajor}) =
    _gemm!('T', 'N', M.α, M.B, transpose(M.A), M.β, transpose(M.C))
@inline materialize!(M::BlasMatMulMatAdd{<:ConjLayout{<:AbstractRowMajor},<:AbstractColumnMajor,<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('C', 'N', M.α, M.B, adjoint(M.A), M.β, adjoint(M.C))


@inline materialize!(M::BlasMatMulMatAdd{<:AbstractRowMajor,<:AbstractRowMajor,<:AbstractRowMajor}) =
    _gemm!('N', 'N', M.α, transpose(M.B), transpose(M.A), M.β, transpose(M.C))
@inline materialize!(M::BlasMatMulMatAdd{<:ConjLayout{<:AbstractRowMajor},<:ConjLayout{<:AbstractRowMajor},<:ConjLayout{<:AbstractRowMajor},<:BlasComplex}) =
    _gemm!('N', 'N', M.α, adjoint(M.B), adjoint(M.A), M.β, adjoint(M.C))


###
# Symmetric
###

# make copy to make sure always works
@inline _symv!(tA, α, A, x, β, y) = BLAS.symv!(tA, α, unalias(y,A), unalias(y,x), β, y)
@inline _hemv!(tA, α, A, x, β, y) = BLAS.hemv!(tA, α, unalias(y,A), unalias(y,x), β, y)


materialize!(M::BlasMatMulVecAdd{<:SymmetricLayout{<:AbstractColumnMajor},<:AbstractStridedLayout,<:AbstractStridedLayout}) =
    _symv!(symmetricuplo(M.A), M.α, symmetricdata(M.A), M.B, M.β, M.C)


materialize!(M::BlasMatMulVecAdd{<:SymmetricLayout{<:AbstractRowMajor},<:AbstractStridedLayout,<:AbstractStridedLayout}) =
    _symv!(symmetricuplo(M.A) == 'L' ? 'U' : 'L', M.α, transpose(symmetricdata(M.A)), M.B, M.β, M.C)


materialize!(M::BlasMatMulVecAdd{<:HermitianLayout{<:AbstractColumnMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasComplex}) =
    _hemv!(symmetricuplo(M.A), M.α, hermitiandata(M.A), M.B, M.β, M.C)

materialize!(M::BlasMatMulVecAdd{<:HermitianLayout{<:AbstractRowMajor},<:AbstractStridedLayout,<:AbstractStridedLayout,<:BlasComplex}) =
    _hemv!(symmetricuplo(M.A) == 'L' ? 'U' : 'L', M.α, hermitiandata(M.A)', M.B, M.β, M.C)


####
# Diagonal
####

# Diagonal multiplication never changes structure
similar(M::MulAdd{<:DiagonalLayout,<:DiagonalLayout}, ::Type{T}, axes) where T = similar(M.B, T, axes)
similar(M::MulAdd{<:DiagonalLayout}, ::Type{T}, axes) where T = similar(M.B, T, axes)
similar(M::MulAdd{<:Any,<:DiagonalLayout}, ::Type{T}, axes) where T = similar(M.A, T, axes)
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


BroadcastStyle(::Type{<:MulAdd}) = ApplyBroadcastStyle()

scalarone(::Type{T}) where T = one(T)
scalarone(::Type{<:AbstractArray{T}}) where T = scalarone(T)
scalarzero(::Type{T}) where T = zero(T)
scalarzero(::Type{<:AbstractArray{T}}) where T = scalarzero(T)

fillzeros(::Type{T}, ax) where T<:Number = Zeros{T}(ax)
mulzeros(::Type{T}, M) where T<:Number = fillzeros(T, axes(M))

# initiate array-valued MulAdd
function _mulzeros!(dest::AbstractVector{T}, A, B) where T
    for k in axes(dest,1)
        dest[k] = similar(Mul(A[k,1],B[1]), eltype(T))
    end
    dest
end

function _mulzeros!(dest::AbstractMatrix{T}, A, B) where T
    for j in axes(dest,2), k in axes(dest,1)
        dest[k,j] = similar(Mul(A[k,1],B[1,j]), eltype(T))
    end
    dest
end

mulzeros(::Type{T}, M) where T<:AbstractArray = _mulzeros!(similar(Array{T}, axes(M)), M.A, M.B)

###
# Fill
###

copy(M::MulAdd{<:AbstractFillLayout,<:AbstractFillLayout,<:AbstractFillLayout}) = M.α*M.A*M.B + M.β*M.C

###
# DualLayout
###

function similar(M::MulAdd{<:DualLayout,<:Any,ZerosLayout}, ::Type{T}, (x,y)) where T
    @assert x ≡ Base.OneTo(1)
    similar(M.A', T, y)'
end