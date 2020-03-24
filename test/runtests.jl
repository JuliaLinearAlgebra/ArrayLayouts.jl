using ArrayLayouts, Test
import ArrayLayouts: MemoryLayout

include("test_layouts.jl")
include("test_muladd.jl")
include("test_ldiv.jl")

struct MyMatrix <: LayoutMatrix{Float64}
    A::Matrix{Float64}
end

Base.getindex(A::MyMatrix, k::Int, j::Int) = A.A[k,j]
Base.size(A::MyMatrix) = size(A.A)
Base.strides(A::MyMatrix) = strides(A.A)
Base.unsafe_convert(::Type{Ptr{T}}, A::MyMatrix) where T = Base.unsafe_convert(Ptr{T}, A.A)
MemoryLayout(::Type{MyMatrix}) = DenseColumnMajor()

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
end