module TestCumsum

using ArrayLayouts, Test

include("infinitearrays.jl")

@testset "RangeCumsum" begin
    @testset for p in (Base.OneTo(5), 2:5, 2:2:6, 6:-2:1, -1.0:3.0:5.0, (-1.0:3.0:5.0)*im,
                        Base.IdentityUnitRange(4:6))
        r = RangeCumsum(p)
        @test parent(r) == p
        @test r == r
        if axes(r,1) isa Base.OneTo
            @test r == cumsum(p)
            @test r .+ 1 == cumsum(p) .+ 1
            @test r[Base.OneTo(3)] == r[1:3]
            @test @view(r[Base.OneTo(3)]) === r[Base.OneTo(3)] == r[1:3]
            @test @view(r[Base.OneTo(3)]) isa RangeCumsum
            @test diff(r) == diff(Vector(r))
            @test -r == -Vector(r)
        end
        @test diff(r) == p[firstindex(p)+1:end]
        @test last(r) == r[end] == sum(p)
        @test first(r) == r[firstindex(r)] == first(p)
    end

    a,b = RangeCumsum(Base.OneTo(5)), RangeCumsum(Base.OneTo(6))
    @test union(a,b) ≡ union(b,a) ≡ b
    @test sort!(copy(a)) == a
    @test sort!(a) ≡ a
    @test sort(a) ≡ a == Vector(a)

    r = RangeCumsum(-4:4)
    @test sort(r) == sort(Vector(r))

    a = RangeCumsum(Base.OneTo(3))
    b = RangeCumsum(1:3)
    @test oftype(a, b) === a

    r = RangeCumsum(InfiniteArrays.OneToInf())
    @test axes(r, 1) == InfiniteArrays.OneToInf()

    @testset "multiplication by a number" begin
        function test_broadcast(n, r)
            w = Vector(r)
            @test n * r isa RangeCumsum
            @test n * r ≈ n * w
            @test r * n isa RangeCumsum
            @test r * n ≈ w * n
        end
        for p in (Base.OneTo(4), -4:4, -4:2:4, -1.0:3.0:5.0)
            r = RangeCumsum(p)
            test_broadcast(3, r)
            test_broadcast(3.5, r)
            test_broadcast(3.5 + 2im, r)
        end
    end
end

end
