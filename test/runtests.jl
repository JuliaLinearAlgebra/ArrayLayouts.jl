using ArrayLayouts, Random, FillArrays, Test, Base64
import ArrayLayouts: MemoryLayout, @_layoutlmul, triangulardata

Random.seed!(0)

include("test_layouts.jl")
include("test_muladd.jl")
include("test_ldiv.jl")

struct MyMatrix <: LayoutMatrix{Float64}
    A::Matrix{Float64}
end

Base.getindex(A::MyMatrix, k::Int, j::Int) = A.A[k,j]
Base.setindex!(A::MyMatrix, v, k::Int, j::Int) = setindex!(A.A, v, k, j)
Base.size(A::MyMatrix) = size(A.A)
Base.strides(A::MyMatrix) = strides(A.A)
Base.unsafe_convert(::Type{Ptr{T}}, A::MyMatrix) where T = Base.unsafe_convert(Ptr{T}, A.A)
MemoryLayout(::Type{MyMatrix}) = DenseColumnMajor()

struct MyVector <: LayoutVector{Float64}
    A::Vector{Float64}
end

Base.getindex(A::MyVector, k::Int) = A.A[k]
Base.setindex!(A::MyVector, v, k::Int) = setindex!(A.A, v, k)
Base.size(A::MyVector) = size(A.A)
Base.strides(A::MyVector) = strides(A.A)
Base.unsafe_convert(::Type{Ptr{T}}, A::MyVector) where T = Base.unsafe_convert(Ptr{T}, A.A)
MemoryLayout(::Type{MyVector}) = DenseColumnMajor()

@testset "LayoutArray" begin
    @testset "LayoutVector" begin
        A = MyVector(randn(5))
        @test A == A.A == Vector(A)
        @test A[1:3] == A.A[1:3]
    end

    @testset "LayoutMatrix" begin
        A = MyMatrix(randn(5,5))
        for (kr,jr) in ((1:2,2:3), (:,:), (:,1:2), (2:3,:), ([1,2],3:4), (:,[1,2]), ([2,3],:))
            @test A[kr,jr] == A.A[kr,jr]
        end
        b = randn(5)
        for Tri in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)
            @test ldiv!(Tri(A), copy(b)) ≈ ldiv!(Tri(A.A), copy(b))
            @test lmul!(Tri(A), copy(b)) ≈ lmul!(Tri(A.A), copy(b))
        end

        @test copyto!(MyMatrix(Array{Float64}(undef,5,5)), A) == A
        @test copyto!(Array{Float64}(undef,5,5), A) == A
        @test copyto!(MyMatrix(Array{Float64}(undef,5,5)), A.A) == A
        @test copyto!(view(MyMatrix(Array{Float64}(undef,5,5)),1:3,1:3), view(A,1:3,1:3)) == A[1:3,1:3]
        @test copyto!(view(MyMatrix(Array{Float64}(undef,5,5)),:,:), A) == A
        @test copyto!(MyMatrix(Array{Float64}(undef,3,3)), view(A,1:3,1:3)) == A[1:3,1:3]
        @test copyto!(view(MyMatrix(Array{Float64}(undef,5,5)),:,:), A.A) == A
        @test copyto!(Array{Float64}(undef,3,3), view(A,1:3,1:3)) == A[1:3,1:3]

        @test qr(A).factors ≈ qr(A.A).factors
        @test qr(A,Val(true)).factors ≈ qr(A.A,Val(true)).factors
        @test lu(A).factors ≈ lu(A.A).factors
        @test lu(A,Val(true)).factors ≈ lu(A.A,Val(true)).factors
        @test_throws ErrorException qr!(A)
        @test_throws ErrorException lu!(A)

        @test qr(A) isa LinearAlgebra.QRCompactWY
        @test inv(A) ≈ inv(A.A)

        Bin = randn(5,5)
        B = MyMatrix(copy(Bin))
        muladd!(1.0, A, A, 2.0, B)
        @test all(B .=== A.A^2 + 2Bin)

        # tiled_blasmul!
        B = MyMatrix(copy(Bin))
        muladd!(1.0, Ones(5,5), A, 2.0, B)

        #generic_blasmul!
        A = BigFloat.(randn(5,5))
        Bin = BigFloat.(randn(5,5))
        B = copy(Bin)
        muladd!(1.0, Ones(5,5), A, 2.0, B)
        @test B == Ones(5,5)*A + 2.0Bin

        C = MyMatrix([1 2; 3 4])
        @test stringmime("text/plain", C) == "2×2 MyMatrix:\n 1.0  2.0\n 3.0  4.0"
    end
end

struct MyUpperTriangular{T} <: AbstractMatrix{T}
    A::UpperTriangular{T,Matrix{T}}
end

MyUpperTriangular{T}(::UndefInitializer, n::Int, m::Int) where T = MyUpperTriangular{T}(UpperTriangular(Array{T}(undef, n, m)))
MyUpperTriangular(A::AbstractMatrix{T}) where T = MyUpperTriangular{T}(UpperTriangular(Matrix{T}(A)))
Base.convert(::Type{MyUpperTriangular{T}}, A::MyUpperTriangular{T}) where T = A
Base.convert(::Type{MyUpperTriangular{T}}, A::MyUpperTriangular) where T = MyUpperTriangular(convert(AbstractArray{T}, A.A))
Base.convert(::Type{MyUpperTriangular}, A::MyUpperTriangular)= A
Base.convert(::Type{AbstractArray{T}}, A::MyUpperTriangular) where T = MyUpperTriangular(convert(AbstractArray{T}, A.A))
Base.convert(::Type{AbstractMatrix{T}}, A::MyUpperTriangular) where T = MyUpperTriangular(convert(AbstractArray{T}, A.A))
Base.convert(::Type{MyUpperTriangular{T}}, A::AbstractArray{T}) where T = MyUpperTriangular{T}(A)
Base.convert(::Type{MyUpperTriangular{T}}, A::AbstractArray) where T = MyUpperTriangular{T}(convert(AbstractArray{T}, A))
Base.convert(::Type{MyUpperTriangular}, A::AbstractArray{T}) where T = MyUpperTriangular{T}(A)
Base.getindex(A::MyUpperTriangular, kj...) = A.A[kj...]
Base.getindex(A::MyUpperTriangular, ::Colon, j::AbstractVector) = MyUpperTriangular(A.A[:,j])
Base.setindex!(A::MyUpperTriangular, v, kj...) = setindex!(A.A, v, kj...)
Base.size(A::MyUpperTriangular) = size(A.A)
Base.similar(::Type{MyUpperTriangular{T}}, m::Int, n::Int) where T = MyUpperTriangular{T}(undef, m, n)
Base.similar(::MyUpperTriangular{T}, m::Int, n::Int) where T = MyUpperTriangular{T}(undef, m, n)
Base.similar(::MyUpperTriangular, ::Type{T}, m::Int, n::Int) where T = MyUpperTriangular{T}(undef, m, n)
LinearAlgebra.factorize(A::MyUpperTriangular) = factorize(A.A)

MemoryLayout(::Type{MyUpperTriangular{T}}) where T = MemoryLayout(UpperTriangular{T,Matrix{T}})
triangulardata(A::MyUpperTriangular) = triangulardata(A.A)

@_layoutlmul MyUpperTriangular


@testset "MyUpperTriangular" begin
    A = randn(5,5)
    B = randn(5,5)
    x = randn(5)
    U = MyUpperTriangular(A)

    @test lmul!(U, copy(x)) ≈ U * x
    @test lmul!(U, copy(B)) ≈ U * B

    @test_skip lmul!(U,view(copy(B),collect(1:5),1:5)) ≈ U * B
end

