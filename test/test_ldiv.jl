using ArrayLayouts, LinearAlgebra, Test
import ArrayLayouts: ApplyBroadcastStyle

@testset "Ldiv" begin
    @testset "Float64 \\ *" begin
        A = randn(5,5)
        b = randn(5)
        M = Ldiv(A,b)

        @test size(M) == (5,)
        @test similar(M) isa Vector{Float64}
        @test materialize(M) isa Vector{Float64}
        @test all(materialize(M) .=== (A\b))

        @test Base.BroadcastStyle(typeof(Ldiv(A,b))) isa ApplyBroadcastStyle

        @test all(copyto!(similar(b), Ldiv(A,b)) .===
                    (similar(b) .= Ldiv(A,b)) .=== materialize(Ldiv(A,b)) .===
                  (A\b) .=== (b̃ =  copy(b); LAPACK.gesv!(copy(A), b̃); b̃))

        @test copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) ≈ UpperTriangular(A) \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A),b)) .===
                    BLAS.trsv('U', 'N', 'N', A, b) )

        @test copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) ≈ UpperTriangular(A)' \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A)',b)) .===
                    copyto!(similar(b), Ldiv(transpose(UpperTriangular(A)) , b)) .===
                            (similar(b) .= Ldiv(transpose(UpperTriangular(A)),b)) .===
                    BLAS.trsv('U', 'T', 'N', A, b))
    end

    @testset "ComplexF64 \\ *" begin
        T = ComplexF64
        A = randn(T,5,5)
        b = randn(T,5)
        @test all(copyto!(similar(b), Ldiv(A,b)) .===
                    (similar(b) .= Ldiv(A,b)) .===
                  (A\b) .=== (b̃ =  copy(b); LAPACK.gesv!(copy(A), b̃); b̃))

        @test copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) ≈ UpperTriangular(A) \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A) , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A),b)) .===
                    BLAS.trsv('U', 'N', 'N', A, b) )

        @test copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) ≈ UpperTriangular(A)' \ b
        @test all(copyto!(similar(b), Ldiv(UpperTriangular(A)' , b)) .===
                    (similar(b) .= Ldiv(UpperTriangular(A)',b)) .===
                    BLAS.trsv('U', 'C', 'N', A, b))

        @test copyto!(similar(b), Ldiv(transpose(UpperTriangular(A)) , b)) ≈ transpose(UpperTriangular(A)) \ b
        @test all(copyto!(similar(b), Ldiv(transpose(UpperTriangular(A)) , b)) .===
                            (similar(b) .= Ldiv(transpose(UpperTriangular(A)),b)) .===
                    BLAS.trsv('U', 'T', 'N', A, b))
    end

    @testset "BigFloat Triangular \\" begin
        for T in (Float64, ComplexF64)
            A = big.(randn(T,5,5))
            b = big.(randn(T,5))
            @test materialize!(Ldiv(UpperTriangular(A),copy(b))) ≈ UpperTriangular(A)\b
            @test materialize!(Ldiv(UnitUpperTriangular(A),copy(b))) ≈ UnitUpperTriangular(A)\b
            @test materialize!(Ldiv(LowerTriangular(A),copy(b))) ≈ LowerTriangular(A)\b
            @test materialize!(Ldiv(UnitLowerTriangular(A),copy(b))) ≈ UnitLowerTriangular(A)\b

            @test materialize!(Ldiv(UpperTriangular(A)',copy(b))) ≈ UpperTriangular(A)'\b
            @test materialize!(Ldiv(UnitUpperTriangular(A)',copy(b))) ≈ UnitUpperTriangular(A)'\b
            @test materialize!(Ldiv(LowerTriangular(A)',copy(b))) ≈ LowerTriangular(A)'\b
            @test materialize!(Ldiv(UnitLowerTriangular(A)',copy(b))) ≈ UnitLowerTriangular(A)'\b

            @test materialize!(Ldiv(transpose(UpperTriangular(A)),copy(b))) ≈ transpose(UpperTriangular(A))\b
            @test materialize!(Ldiv(transpose(UnitUpperTriangular(A)),copy(b))) ≈ transpose(UnitUpperTriangular(A))\b
            @test materialize!(Ldiv(transpose(LowerTriangular(A)),copy(b))) ≈ transpose(LowerTriangular(A))\b
            @test materialize!(Ldiv(transpose(UnitLowerTriangular(A)),copy(b))) ≈ transpose(UnitLowerTriangular(A))\b

            B = big.(randn(T,5,5))
            @test materialize!(Ldiv(UpperTriangular(A),copy(B))) ≈ UpperTriangular(A)\B
        end
    end

    @testset "Triangular \\ matrix" begin
        A = randn(5,5)
        b = randn(5,5)
        M =  Ldiv(UpperTriangular(A), b)
        @test Base.Broadcast.broadcastable(M) === M
        @test UpperTriangular(A) \ b ≈ copyto!(similar(b) , Ldiv(UpperTriangular(A), b)) ≈
            (b .= Ldiv(UpperTriangular(A), b))
    end

    @testset "Int" begin
        A = [1 2 ; 3 4]; b = [5,6];
        @test eltype(Ldiv(A, b)) == Float64
    end
end
