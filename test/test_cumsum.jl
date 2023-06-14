using ArrayLayouts, Test

@testset "RangeCumsum" begin
    for r in (RangeCumsum(Base.OneTo(5)), RangeCumsum(2:5), RangeCumsum(2:2:6), RangeCumsum(6:-2:1))
        @test r == cumsum(r.range)
        @test r == r
        @test r .+ 1 == cumsum(r.range) .+ 1
        @test r[Base.OneTo(3)] == r[1:3]
        @test last(r) == r[end]
        @test diff(r) == diff(Vector(r))
        @test first(r) == r[1]
    end

    a,b = RangeCumsum(Base.OneTo(5)), RangeCumsum(Base.OneTo(6))
    @test union(a,b) ≡ union(b,a) ≡ b
    @test sort!(a) ≡ a

    a = RangeCumsum(Base.OneTo(3))
    b = RangeCumsum(1:3)
    @test oftype(a, b) === a
end
