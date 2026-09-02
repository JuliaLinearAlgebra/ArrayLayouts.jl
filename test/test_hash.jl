module TestHash

using ArrayLayouts, LinearAlgebra, Test, Infinities

include("infinitearrays.jl")
using .InfiniteArrays

@testset "hash" begin
    @testset "finite arrays hash as in Base" begin
        for (a, b) in ((RangeCumsum(Base.OneTo(4)), [1,3,6,10]),
                       (RangeCumsum(2:5), [2,5,9,14]),
                       (Diagonal(RangeCumsum(Base.OneTo(3))), Diagonal([1,3,6])),
                       (RangeCumsum(Base.OneTo(3))', [1 3 6]))
            @test hash(a) == hash(b)
            @test hash(a, UInt(7)) == hash(b, UInt(7))
        end
        @test hash(RangeCumsum(Base.OneTo(3))) == hash(RangeCumsum(1:3))
    end

    @testset "infinite arrays" begin
        r = RangeCumsum(OneToInf())
        @test hash(r) isa UInt
        @test hash(r) == hash(RangeCumsum(OneToInf{Int16}()))
        @test hash(r, UInt(7)) == hash(RangeCumsum(OneToInf{Int16}()), UInt(7))
        @test hash(r) ≠ hash(RangeCumsum(InfiniteArrays.InfUnitRange(2)))
        @test isequal(r, RangeCumsum(OneToInf()))

        # a `LayoutArray` whose entries are only realised on demand
        v = InfiniteArrays.InfVec()
        @test hash(v) isa UInt
        @test hash(v) == hash(v)
        @test hash(v) ≠ hash(InfiniteArrays.InfVec())   # distinct data
        A = InfiniteArrays.InfMat()
        @test hash(A) isa UInt
        @test hash(A) == hash(A)

        @testset "wrappers" begin
            @test hash(v') == hash(transpose(v))
            @test hash(Diagonal(v)) isa UInt
            @test hash(Diagonal(v)) == hash(Diagonal(v))
            @test hash(InfBidiagonal(:U)) isa UInt
            @test hash(InfUpperTriangular()) isa UInt
            @test hash(Symmetric(A)) isa UInt
            @test hash(view(A, OneToInf(), OneToInf())) isa UInt

            D = Diagonal(v)
            @test hash(D') == hash(D)
        end
    end
end

end # module
