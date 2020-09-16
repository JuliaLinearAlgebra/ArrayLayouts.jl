using ArrayLayouts, FillArrays, Random, Test
import ArrayLayouts: DenseColumnMajor, AbstractStridedLayout, AbstractColumnMajor, DiagonalLayout, mul, Mul, zero!

Random.seed!(0)
@testset "Multiplication" begin
    @testset "zero!" begin
        for A in (randn(5,5), [randn(5,5),randn(4,4)])
            zero!(A)
            @test iszero(A)
        end
    end

    @testset "Matrix * Vector" begin
        @testset "eltype" begin
            v = MulAdd(1,zeros(Int,2,2), zeros(Float64,2),0,zeros(2))
            A = MulAdd(1,zeros(Int,2,2), zeros(Float64,2,2), 0,zeros(2,2))
            @test eltype(v) == eltype(A) == Float64
            @test @inferred(axes(v)) == (@inferred(axes(v,1)),) == (Base.OneTo(2),)
            @test @inferred(size(v)) == (@inferred(size(v,1)),) == (2,)
            @test @inferred(axes(A)) == (@inferred(axes(A,1)),@inferred(axes(A,2))) == (Base.OneTo(2),Base.OneTo(2))
            @test @inferred(size(A)) == (@inferred(size(A,1)),size(A,2)) == (2,2)

            Ã = materialize(A)
            @test Ã isa Matrix{Float64}
            @test Ã == zeros(2,2)

            c = similar(v)
            fill!(c,NaN)
            @test copyto!(c,v) == zeros(2)
            fill!(c,NaN)
            c .= v
            @test c == zeros(2)
        end

        @testset "gemv Float64" begin
            for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                    view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
                b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9))
                c = similar(b);

                muladd!(1.0,A,b,0.0,c)
                @test all(c .=== A*b .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))
                copyto!(c, MulAdd(1.0,A,b,0.0,c))
                @test all(c .=== A*b .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))
                c .= MulAdd(1.0,A,b,0.0,c)
                @test all(c .=== A*b .=== BLAS.gemv!('N', 1.0, A, b, 0.0, similar(c)))

                M = MulAdd(2.0, A,b, 0.0, c)
                @test M isa MulAdd{<:AbstractColumnMajor,<:AbstractStridedLayout,<:AbstractColumnMajor}
                c .= M
                @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 0.0, similar(c)))
                copyto!(c, M)
                @test all(c .=== BLAS.gemv!('N', 2.0, A, b, 0.0, similar(c)))

                c = copy(b)
                M = MulAdd(1.0, A,b, 1.0, c)
                @test M isa MulAdd{<:AbstractColumnMajor,<:AbstractStridedLayout,<:AbstractColumnMajor}
                copyto!(c, M)
                @test all(c .=== BLAS.gemv!('N', 1.0, A, b, 1.0, copy(b)))
            end
        end

        @testset "gemv mixed array types" begin
            (A, b, c) = (randn(5,5), randn(5), 1.0:5.0)
            d = similar(b)
            d .= MulAdd(1.0,A,b,1.0,c)
            @test all(d .=== BLAS.gemv!('N', 1.0, A, b, 1.0, Vector{Float64}(c)))
        end

        @testset "gemv Complex" begin
            for T in (ComplexF64,),
                    A in (randn(T,5,5), view(randn(T,5,5),:,:)),
                    b in (randn(T,5), view(randn(T,5),:))
                c = similar(b);

                c .= MulAdd(one(T), A, b, zero(T), c)
                @test all(c .=== BLAS.gemv!('N', one(T), A, b, zero(T), similar(c)))

                c = copy(b)
                muladd!(2one(T),A,b,one(T),c)
                @test all(c .=== BLAS.gemv!('N', 2one(T), A, b, one(T), copy(b)))
            end
        end

        @testset "Matrix{Int} * Vector{Vector{Int}}" begin
            A, x =  [1 2; 3 4] , [[1,2],[3,4]]
            X = reshape([[1 2],[3 4], [5 6], [7 8]],2,2)
            @test mul(A,x) == A*x
            @test mul(A,X) == A*X
            @test mul(X,A) == X*A
        end
    end

    @testset "Matrix * Matrix" begin
        @testset "gemm" begin
            for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                    view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
                B in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                        view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5))
                C = similar(B);

                C .= MulAdd(1.0,A,B,0.0,C)
                @test all(C .=== BLAS.gemm!('N', 'N', 1.0, A, B, 0.0, similar(C)))

                C .= MulAdd(2.0,A,B,0.0,C)
                @test all(C .=== BLAS.gemm!('N', 'N', 2.0, A, B, 0.0, similar(C)))

                C = copy(B)
                C .= MulAdd(2.0,A,B,1.0,C)
                @test all(C .=== BLAS.gemm!('N', 'N', 2.0, A, B, 1.0, copy(B)))
            end
        end

        @testset "gemm Complex" begin
            for T in (ComplexF64,),
                    A in (randn(T,5,5), view(randn(T,5,5),:,:)),
                    B in (randn(T,5,5), view(randn(T,5,5),:,:))
                C = similar(B);

                C .= MulAdd(one(T),A,B,zero(T),C)
                @test all(C .=== BLAS.gemm!('N', 'N', one(T), A, B, zero(T), similar(C)))

                C .= MulAdd(2one(T),A,B,zero(T),C)
                @test all(C .=== BLAS.gemm!('N', 'N', 2one(T), A, B, zero(T), similar(C)))

                C = copy(B)
                C .= MulAdd(one(T),A,B,one(T),C)
                @test all(C .=== BLAS.gemm!('N', 'N', one(T), A, B, one(T), copy(B)))
            end
        end

        @testset "gemm mixed array types" begin
            (A, B, C) = (randn(5,5), randn(5,5), reshape(1.0:25.0,5,5))
            D = similar(B)
            D .= MulAdd(1.0,A,B,1.0, C)
            @test all(D .=== BLAS.gemm!('N', 'N', 1.0, A, B, 1.0, Matrix{Float64}(C)))
        end
    end

    @testset "adjtrans" begin
        @testset "gemv adjtrans" begin
            for A in (randn(5,5), view(randn(5,5),:,:), view(randn(5,5),1:5,:),
                    view(randn(5,5),1:5,1:5), view(randn(5,5),:,1:5)),
                        b in (randn(5), view(randn(5),:), view(randn(5),1:5), view(randn(9),1:2:9)),
                        Ac in (transpose(A), A')
                c = similar(b);

                c .= MulAdd(1.0,Ac,b,0.0,c)
                @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 0.0, similar(c)))

                c .= MulAdd(2.0,Ac,b,0.0,c)
                @test all(c .=== BLAS.gemv!('T', 2.0, A, b, 0.0, similar(c)))

                c = copy(b)
                c .= MulAdd(1.0,Ac,b,1.0,c)
                @test all(c .=== BLAS.gemv!('T', 1.0, A, b, 1.0, copy(b)))
            end
        end

        @testset "gemv adjtrans mixed types" begin
            (A, b, c) = (randn(5,5), randn(5), 1.0:5.0)
            for Ac in (transpose(A), A')
                d = similar(b)
                d .= MulAdd(1.0,Ac,b,1.0, c)
                @test all(d .=== BLAS.gemv!('T', 1.0, A, b, 1.0, Vector{Float64}(c)))

                d .= MulAdd(3.0,Ac,b,2.0, c)
                @test all(d .=== BLAS.gemv!('T', 3.0, A, b, 2.0, Vector{Float64}(c)))
            end
        end

        @testset "gemv adjtrans Complex" begin
            for T in (ComplexF64,),
                    A in (randn(T,5,5), view(randn(T,5,5),:,:)),
                    b in (randn(T,5), view(randn(T,5),:)),
                    (Ac,trans) in ((transpose(A),'T'), (A','C'))
                c = similar(b);

                c .= MulAdd(one(T),Ac,b,zero(T),c)
                @test all(c .=== BLAS.gemv!(trans, one(T), A, b, zero(T), similar(c)))

                c = copy(b)
                c .= MulAdd(3one(T),Ac,b,2one(T),c)
                @test all(c .=== BLAS.gemv!(trans, 3one(T), A, b, 2one(T), copy(b)))
            end

            C = randn(6,6) + im*randn(6,6)
            V = view(C', 2:3, 3:4)
            c = randn(2) + im*randn(2)
            @test all(muladd!(1.0+0im,V,c,0.0+0im,similar(c,2)) .=== BLAS.gemv!('C', 1.0+0im, Matrix(V'), c, 0.0+0im, similar(c,2)))
            @test all(muladd!(1.0+0im,V',c,0.0+0im,similar(c,2)) .=== BLAS.gemv!('N', 1.0+0im, Matrix(V'), c, 0.0+0im, similar(c,2)))
        end

        @testset "gemm adjtrans" begin
            for A in (randn(5,5), view(randn(5,5),:,:)),
                B in (randn(5,5), view(randn(5,5),1:5,:))
                for Ac in (transpose(A), A')
                    C = copy(B)
                    C .= MulAdd(3.0, Ac, B, 2.0, C)
                    @test all(C .=== BLAS.gemm!('T', 'N', 3.0, A, B, 2.0, copy(B)))
                end
                for Bc in (transpose(B), B')
                    C = copy(B)
                    C .= MulAdd(3.0, A, Bc, 2.0, C)
                    @test all(C .=== BLAS.gemm!('N', 'T', 3.0, A, B, 2.0, copy(B)))
                end
                for Ac in (transpose(A), A'), Bc in (transpose(B), B')
                    C = copy(B)
                    d = similar(C)
                    d .= MulAdd(3.0, Ac, Bc, 2.0, C)
                    @test all(d .=== BLAS.gemm!('T', 'T', 3.0, A, B, 2.0, copy(B)))
                end
            end
        end

        @testset "symv adjtrans" begin
            A = randn(100,100)
            x = randn(100)
            for As in (Symmetric(A), Hermitian(A), Symmetric(A)', Symmetric(view(A,:,:)',:L), view(Symmetric(A),:,:))
                @test all( muladd!(1.0,As,x,0.0,similar(x)) .=== BLAS.symv!('U', 1.0, A, x, 0.0, similar(x)) )
            end

            for As in (Symmetric(A,:L), Hermitian(A,:L), Symmetric(A,:L)', Symmetric(view(A,:,:)',:U), view(Hermitian(A,:L),:,:))
                @test all( muladd!(1.0,As,x,0.0,similar(x)) .=== BLAS.symv!('L', 1.0, A, x, 0.0, similar(x)) )
            end

            T = ComplexF64
            A = randn(T,100,100)
            x = randn(T,100)
            for As in (Symmetric(A), transpose(Symmetric(A)), Symmetric(transpose(view(A,:,:)),:L), view(Symmetric(A),:,:))
                @test all( muladd!(one(T),As,x,zero(T),similar(x)) .=== BLAS.symv!('U', one(T), A, x, zero(T), similar(x)) )
            end

            for As in (Symmetric(A,:L), transpose(Symmetric(A,:L)), Symmetric(transpose(view(A,:,:)),:U), view(Symmetric(A,:L),:,:))
                @test all( muladd!(one(T),As,x,zero(T),similar(x)) .=== BLAS.symv!('L', one(T), A, x, zero(T), similar(x)) )
            end

            for As in(Hermitian(A), Hermitian(A)', view(Hermitian(A),:,:))
                @test all( muladd!(one(T),As,x,zero(T),similar(x)) .=== BLAS.hemv!('U', one(T), A, x, zero(T), similar(x)) )
            end

            for As in(Hermitian(A,:L), Hermitian(A,:L)', view(Hermitian(A,:L),:,:))
                @test all( muladd!(one(T),As,x,zero(T),similar(x)) .=== BLAS.hemv!('L', one(T), A, x, zero(T), similar(x)) )
            end
        end
    end

    @testset "Mul" begin
        @testset "Mixed types" begin
            A = randn(5,6)
            b = rand(Int,6)
            c = Array{Float64}(undef, 5)
            d = similar(c)
            copyto!(d, MulAdd(1, A, b, 0.0, d))
            @test d == copyto!(similar(d), MulAdd(1, A, b, 0.0, d)) ≈ A*b
            @test copyto!(similar(d), MulAdd(1, A, b, 1.0, d)) ≈ A*b + d

            @test all((similar(d) .= MulAdd(1, A, b, 1.0, d)) .=== copyto!(similar(d), MulAdd(1, A, b, 1.0, d)))

            B = rand(Int,6,4)
            C = Array{Float64}(undef, 5, 4)

            @test_throws DimensionMismatch muladd!(1.0,A,B,0.0,similar(C,4,4))

            A = randn(Float64,20,22)
            B = randn(ComplexF64,22,24)
            C = similar(B,20,24)
            @test copyto!(similar(C), MulAdd(1.0, A, B, 0.0, C)) ≈ A*B
        end

        @testset "no allocation" begin        
            A = randn(5,5); x = randn(5); y = randn(5); c = similar(y);
            muladd!(2.0, A, x, 3.0, y)
            @test @allocated(muladd!(2.0, A, x, 3.0, y)) == 0
            VA = view(A,:,1:2)
            vx = view(x,1:2)
            vy = view(y,:)
            muladd!(2.0, VA, vx, 3.0, vy)
            @test @allocated(muladd!(2.0, VA, vx, 3.0, vy)) == 0
        end

        @testset "BigFloat" begin
            A = BigFloat.(randn(5,5))
            x = BigFloat.(randn(5))
            @test A*x == muladd!(1.0,A,x,0.0,copy(x))
            @test_throws UndefRefError muladd!(1.0,A,x,0.0,similar(x))
        end

        @testset "Scalar * Vector" begin
            A, x =  [1 2; 3 4] , [[1,2],[3,4]]
            @test muladd!(1.0,A,x,0.0, 1.0*x) == A*x
        end 

        @testset "adjoint" begin
            A = randn(5,5)
            V = view(A',1:3,1:3)
            b = randn(3)
            @test MemoryLayout(V) isa RowMajor
            @test MemoryLayout(V') isa ColumnMajor
            @test strides(V) == (5,1)
            @test strides(V') == (1,5)
            @test mul(V, b) ≈ V*b ≈ Matrix(V)*b
            @test mul(V', b) ≈ V'b ≈ Matrix(V)'*b
        end
    end

    @testset "Lmul/Rmul" begin
        @testset "tri Lmul" begin
            @testset "Float * Float vector" begin
                A = randn(Float64, 100, 100)
                x = randn(Float64, 100)

                L = Lmul(UpperTriangular(A),x)
                @test size(L) == (size(L,1),) == (100,)
                @test axes(L) == (axes(L,1),) == (Base.OneTo(100),)
                @test eltype(L) == Float64
                @test length(L) == 100

                @test similar(L) isa Vector{Float64}
                @test similar(L,Int) isa Vector{Int}
                
                @test all(mul(UpperTriangular(A),x) .=== 
                            copy(Lmul(UpperTriangular(A),x)) .===
                            ArrayLayouts.lmul!(UpperTriangular(A),copy(x)) .===
                            copyto!(similar(x),Lmul(UpperTriangular(A),x)) .===
                            UpperTriangular(A)*x .===
                            BLAS.trmv!('U', 'N', 'N', A, copy(x)))
                @test all(copyto!(similar(x),Lmul(UnitUpperTriangular(A),x)) .=== 
                            UnitUpperTriangular(A)*x .===
                            BLAS.trmv!('U', 'N', 'U', A, copy(x)))
                @test all(copyto!(similar(x),Lmul(LowerTriangular(A),x)) .=== 
                            LowerTriangular(A)*x .===
                            BLAS.trmv!('L', 'N', 'N', A, copy(x)))
                @test all(ArrayLayouts.lmul!(UnitLowerTriangular(A),copy(x)) .===
                            UnitLowerTriangular(A)x .===
                            BLAS.trmv!('L', 'N', 'U', A, copy(x)))

                @test all(ArrayLayouts.lmul!(UpperTriangular(A)',copy(x)) .===
                            ArrayLayouts.lmul!(LowerTriangular(A'),copy(x)) .===
                            UpperTriangular(A)'x .===
                            BLAS.trmv!('U', 'T', 'N', A, copy(x)))
                @test all(ArrayLayouts.lmul!(UnitUpperTriangular(A)',copy(x)) .===
                            ArrayLayouts.lmul!(UnitLowerTriangular(A'),copy(x)) .===
                            UnitUpperTriangular(A)'*x .===
                            BLAS.trmv!('U', 'T', 'U', A, copy(x)))
                @test all(ArrayLayouts.lmul!(LowerTriangular(A)',copy(x)) .===
                            ArrayLayouts.lmul!(UpperTriangular(A'),copy(x)) .===
                            LowerTriangular(A)'*x .===
                            BLAS.trmv!('L', 'T', 'N', A, copy(x)))
                @test all(ArrayLayouts.lmul!(UnitLowerTriangular(A)',copy(x)) .===
                            ArrayLayouts.lmul!(UnitUpperTriangular(A'),copy(x)) .===
                            UnitLowerTriangular(A)'*x .===
                            BLAS.trmv!('L', 'T', 'U', A, copy(x)))

                @test_throws DimensionMismatch mul(UpperTriangular(A), randn(5))
                @test_throws DimensionMismatch ArrayLayouts.lmul!(UpperTriangular(A), randn(5))
            end

            @testset "Float * Complex vector"  begin
                T = ComplexF64
                A = randn(T, 100, 100)
                x = randn(T, 100)

                @test all((y = copy(x); y .= Lmul(UpperTriangular(A),y) ) .===
                            UpperTriangular(A)x .===
                            BLAS.trmv!('U', 'N', 'N', A, copy(x)))
                @test all((y = copy(x); y .= Lmul(UnitUpperTriangular(A),y) ) .===
                            UnitUpperTriangular(A)x .===
                            BLAS.trmv!('U', 'N', 'U', A, copy(x)))
                @test all((y = copy(x); y .= Lmul(LowerTriangular(A),y) ) .===
                            LowerTriangular(A)x .===
                            BLAS.trmv!('L', 'N', 'N', A, copy(x)))
                @test all((y = copy(x); y .= Lmul(UnitLowerTriangular(A),y) ) .===
                            UnitLowerTriangular(A)x .===
                            BLAS.trmv!('L', 'N', 'U', A, copy(x)))
                @test LowerTriangular(A')  == UpperTriangular(A)'

                @test all((y = copy(x); y .= Lmul(transpose(UpperTriangular(A)),y) ) .===
                            (similar(x) .= Lmul(LowerTriangular(transpose(A)),x)) .===
                            transpose(UpperTriangular(A))x .===
                            BLAS.trmv!('U', 'T', 'N', A, copy(x)))
                @test all((y = copy(x); y .= Lmul(transpose(UnitUpperTriangular(A)),y) ) .===
                            (similar(x) .= Lmul(UnitLowerTriangular(transpose(A)),x)) .===
                            transpose(UnitUpperTriangular(A))x .===
                            BLAS.trmv!('U', 'T', 'U', A, copy(x)))
                @test all((y = copy(x); y .= Lmul(transpose(LowerTriangular(A)),y) ) .===
                            (similar(x) .= Lmul(UpperTriangular(transpose(A)),x)) .===
                            transpose(LowerTriangular(A))x .===
                            BLAS.trmv!('L', 'T', 'N', A, copy(x)))
                @test all((y = copy(x); y .= Lmul(transpose(UnitLowerTriangular(A)),y) ) .===
                            (similar(x) .= Lmul(UnitUpperTriangular(transpose(A)),x)) .===
                            transpose(UnitLowerTriangular(A))x .===
                            BLAS.trmv!('L', 'T', 'U', A, copy(x)))

                @test all((y = copy(x); y .= Lmul(UpperTriangular(A)',y) ) .===
                            (similar(x) .= Lmul(LowerTriangular(A'),x)) .===
                            UpperTriangular(A)'x .===
                            BLAS.trmv!('U', 'C', 'N', A, copy(x)))
                @test all((y = copy(x); y .= Lmul(UnitUpperTriangular(A)',y) ) .===
                            (similar(x) .= Lmul(UnitLowerTriangular(A'),x)) .===
                            UnitUpperTriangular(A)'x .===
                            BLAS.trmv!('U', 'C', 'U', A, copy(x)))
                @test all((y = copy(x); y .= Lmul(LowerTriangular(A)',y) ) .===
                            (similar(x) .= Lmul(UpperTriangular(A'),x)) .===
                            LowerTriangular(A)'x .===
                            BLAS.trmv!('L', 'C', 'N', A, copy(x)))
                @test all((y = copy(x); y .= Lmul(UnitLowerTriangular(A)',y) ) .===
                            (similar(x) .= Lmul(UnitUpperTriangular(A'),x)) .===
                            UnitLowerTriangular(A)'x .===
                            BLAS.trmv!('L', 'C', 'U', A, copy(x)))
            end

            @testset "Float * Float Matrix" begin
                A = randn(Float64, 100, 100)
                x = randn(Float64, 100, 100)

                @test UpperTriangular(A)*x ≈ (similar(x) .= Lmul(UpperTriangular(A), x))
            end

            @testset "adjtrans" begin
                for T in (Float64, ComplexF64)
                    A = randn(T,100,100)
                    b = randn(T,100)

                    @test all(copy(Lmul(UpperTriangular(A)', b)) .=== UpperTriangular(A)'b)
                    @test all(copy(Lmul(UnitUpperTriangular(A)', b)) .=== UnitUpperTriangular(A)'b)
                    @test all(copy(Lmul(LowerTriangular(A)', b)) .=== LowerTriangular(A)'b)
                    @test all(copy(Lmul(UnitLowerTriangular(A)', b)) .=== UnitLowerTriangular(A)'b)

                    @test all(copy(Lmul(transpose(UpperTriangular(A)), b)) .=== transpose(UpperTriangular(A))b)
                    @test all(copy(Lmul(transpose(UnitUpperTriangular(A)), b)) .=== transpose(UnitUpperTriangular(A))b)
                    @test all(copy(Lmul(transpose(LowerTriangular(A)), b)) .=== transpose(LowerTriangular(A))b)
                    @test all(copy(Lmul(transpose(UnitLowerTriangular(A)), b)) .=== transpose(UnitLowerTriangular(A))b)

                    B = randn(T,100,100)

                    @test all(copy(Lmul(UpperTriangular(A)', B)) .=== UpperTriangular(A)'B)
                    @test all(copy(Lmul(UnitUpperTriangular(A)', B)) .=== UnitUpperTriangular(A)'B)
                    @test all(copy(Lmul(LowerTriangular(A)', B)) .=== LowerTriangular(A)'B)
                    @test all(copy(Lmul(UnitLowerTriangular(A)', B)) .=== UnitLowerTriangular(A)'B)

                    @test all(copy(Lmul(transpose(UpperTriangular(A)), B)) .=== transpose(UpperTriangular(A))B)
                    @test all(copy(Lmul(transpose(UnitUpperTriangular(A)), B)) .=== transpose(UnitUpperTriangular(A))B)
                    @test all(copy(Lmul(transpose(LowerTriangular(A)), B)) .=== transpose(LowerTriangular(A))B)
                    @test all(copy(Lmul(transpose(UnitLowerTriangular(A)), B)) .=== transpose(UnitLowerTriangular(A))B)                
                end

                for T in (Float64, ComplexF64)
                    A = big.(randn(T,100,100))
                    b = big.(randn(T,100))

                    @test all(copy(Lmul(UpperTriangular(A)', b)) ≈ UpperTriangular(A)'b)
                    @test all(copy(Lmul(UnitUpperTriangular(A)', b)) ≈ UnitUpperTriangular(A)'b)
                    @test all(copy(Lmul(LowerTriangular(A)', b)) ≈ LowerTriangular(A)'b)
                    @test all(copy(Lmul(UnitLowerTriangular(A)', b)) ≈ UnitLowerTriangular(A)'b)

                    @test all(copy(Lmul(transpose(UpperTriangular(A)), b)) ≈ transpose(UpperTriangular(A))b)
                    @test all(copy(Lmul(transpose(UnitUpperTriangular(A)), b)) ≈ transpose(UnitUpperTriangular(A))b)
                    @test all(copy(Lmul(transpose(LowerTriangular(A)), b)) ≈ transpose(LowerTriangular(A))b)
                    @test all(copy(Lmul(transpose(UnitLowerTriangular(A)), b)) ≈ transpose(UnitLowerTriangular(A))b)

                    B = big.(randn(T,100,100))
                    
                    @test all(copy(Lmul(UpperTriangular(A)', B)) ≈ UpperTriangular(A)'B)
                    @test all(copy(Lmul(UnitUpperTriangular(A)', B)) ≈ UnitUpperTriangular(A)'B)
                    @test all(copy(Lmul(LowerTriangular(A)', B)) ≈ LowerTriangular(A)'B)
                    @test all(copy(Lmul(UnitLowerTriangular(A)', B)) ≈ UnitLowerTriangular(A)'B)

                    @test all(copy(Lmul(transpose(UpperTriangular(A)), B)) ≈ transpose(UpperTriangular(A))B)
                    @test all(copy(Lmul(transpose(UnitUpperTriangular(A)), B)) ≈ transpose(UnitUpperTriangular(A))B)
                    @test all(copy(Lmul(transpose(LowerTriangular(A)), B)) ≈ transpose(LowerTriangular(A))B)
                    @test all(copy(Lmul(transpose(UnitLowerTriangular(A)), B)) ≈ transpose(UnitLowerTriangular(A))B)                
                end
            end

            @testset "BigFloat" begin
                A = BigFloat.(randn(10,10))
                b = BigFloat.(randn(10))
                @test mul(UpperTriangular(A), b) == UpperTriangular(A)*b
                @test mul(UnitUpperTriangular(A), b) == UnitUpperTriangular(A)*b
                @test mul(LowerTriangular(A), b) == LowerTriangular(A)*b
                @test mul(UnitLowerTriangular(A), b) == UnitLowerTriangular(A)*b
                @test_throws DimensionMismatch mul(UpperTriangular(A), randn(5))
                @test_throws DimensionMismatch ArrayLayouts.lmul!(UpperTriangular(A), randn(5))
                @test_throws DimensionMismatch ArrayLayouts.lmul!(UnitUpperTriangular(A), randn(5))
                @test_throws DimensionMismatch ArrayLayouts.lmul!(LowerTriangular(A), randn(5))
                @test_throws DimensionMismatch ArrayLayouts.lmul!(UnitLowerTriangular(A), randn(5))
            end
        end

        @testset "tri Rmul" begin
            for T in (Float64, ComplexF64)
                A = randn(T, 100,100)
                B = randn(T, 100,100)
                R = Rmul(copy(A), UpperTriangular(B))
                @test size(R) == (size(R,1),size(R,2)) == (100,100)
                @test axes(R) == (axes(R,1),axes(R,2)) == (Base.OneTo(100),Base.OneTo(100))
                @test eltype(R) == T
                @test length(R) == 100^2

                @test similar(R) isa Matrix{T}
                @test similar(R,Int) isa Matrix{Int}
                
                R2 = deepcopy(R)
                @test all(mul(A, UpperTriangular(B)) .=== BLAS.trmm('R', 'U', 'N', 'N', one(T), B, A) .=== copyto!(similar(R2), R2) .=== materialize!(R))
                @test R.A ≠ A
                @test all(BLAS.trmm('R', 'U', 'T', 'N', one(T), B, A) .=== copy(Rmul(A, transpose(UpperTriangular(B)))) .=== A*transpose(UpperTriangular(B)))
                @test all(BLAS.trmm('R', 'U', 'N', 'U', one(T), B, A) .=== copy(Rmul(A, UnitUpperTriangular(B))).=== A*UnitUpperTriangular(B))
                @test all(BLAS.trmm('R', 'U', 'T', 'U', one(T), B, A) .=== copy(Rmul(A, transpose(UnitUpperTriangular(B)))) .=== A*transpose(UnitUpperTriangular(B)))
                @test all(BLAS.trmm('R', 'L', 'N', 'N', one(T), B, A) .=== copy(Rmul(A, LowerTriangular(B))).=== A*LowerTriangular(B))
                @test all(BLAS.trmm('R', 'L', 'T', 'N', one(T), B, A) .=== copy(Rmul(A, transpose(LowerTriangular(B)))) .=== A*transpose(LowerTriangular(B)))
                @test all(BLAS.trmm('R', 'L', 'N', 'U', one(T), B, A) .=== copy(Rmul(A, UnitLowerTriangular(B))).=== A*UnitLowerTriangular(B))
                @test all(BLAS.trmm('R', 'L', 'T', 'U', one(T), B, A) .=== copy(Rmul(A, transpose(UnitLowerTriangular(B)))) .=== A*transpose(UnitLowerTriangular(B)))

                if T <: Complex
                    @test all(BLAS.trmm('R', 'U', 'C', 'N', one(T), B, A) .=== copy(Rmul(A, UpperTriangular(B)')) .=== A*UpperTriangular(B)')
                    @test all(BLAS.trmm('R', 'U', 'C', 'U', one(T), B, A) .=== copy(Rmul(A, UnitUpperTriangular(B)')) .=== A*UnitUpperTriangular(B)')
                    @test all(BLAS.trmm('R', 'L', 'C', 'N', one(T), B, A) .=== copy(Rmul(A, LowerTriangular(B)')) .=== A*LowerTriangular(B)')
                    @test all(BLAS.trmm('R', 'L', 'C', 'U', one(T), B, A) .=== copy(Rmul(A, UnitLowerTriangular(B)')) .=== A*UnitLowerTriangular(B)')
                end
            end

            T = Float64
            A = big.(randn(100,100))
            B = big.(randn(100,100))
            @test ArrayLayouts.rmul!(copy(A),UpperTriangular(B)) == A*UpperTriangular(B)
        end

        @testset "Diagonal" begin
            A = randn(5,5)
            B = Diagonal(randn(5))
            @test MemoryLayout(B) == DiagonalLayout{DenseColumnMajor}()
            
            @test A*B == ArrayLayouts.rmul!(copy(A),B) == mul(A,B)
            @test B*A == ArrayLayouts.lmul!(B,copy(A)) == mul(B,A) 
            @test B*B == ArrayLayouts.lmul!(B, copy(B)) == mul(B, B)
        end

        @testset "tri * tri" begin
            A = randn(5,5)
            B = randn(5,5)
            @test UpperTriangular(A) * UpperTriangular(B) ≈ mul(UpperTriangular(A) , UpperTriangular(B))
            @test LowerTriangular(A) * LowerTriangular(B) ≈ mul(LowerTriangular(A) , LowerTriangular(B))
            @test LowerTriangular(A) * UpperTriangular(B) ≈ mul(LowerTriangular(A) , UpperTriangular(B))
            @test UnitUpperTriangular(A) * UnitUpperTriangular(B) ≈ mul(UnitUpperTriangular(A) , UnitUpperTriangular(B))
            @test UnitLowerTriangular(A) * UnitLowerTriangular(B) ≈ mul(UnitLowerTriangular(A) , UnitLowerTriangular(B))
            @test UnitLowerTriangular(A) * UnitUpperTriangular(B) ≈ mul(UnitLowerTriangular(A) , UnitUpperTriangular(B))
            @test UnitUpperTriangular(A) * UpperTriangular(B) ≈ mul(UnitUpperTriangular(A) , UpperTriangular(B))
            @test @inferred(mul(UpperTriangular(A) , UpperTriangular(B))) isa UpperTriangular
            @test @inferred(mul(LowerTriangular(A) , LowerTriangular(B))) isa LowerTriangular
            @test @inferred(mul(UnitUpperTriangular(A) , UnitUpperTriangular(B))) isa UnitUpperTriangular
            @test @inferred(mul(UnitLowerTriangular(A) , UnitLowerTriangular(B))) isa UnitLowerTriangular
            @test @inferred(mul(UpperTriangular(A) , UnitUpperTriangular(B))) isa UpperTriangular
            @test @inferred(mul(LowerTriangular(A), UpperTriangular(B))) isa Matrix
            @test @inferred(mul(UnitUpperTriangular(A), LowerTriangular(B))) isa Matrix
        end

        @testset "diag * tri" begin
            D = Diagonal(randn(5))
            U = UpperTriangular(randn(5,5))
            @test mul(D,U) == D*U
            @test mul(U,D) == U*D
        end
    end

    @testset "MulAdd" begin
        A = randn(5,5)
        B = randn(5,4)
        C = randn(5,4)
        b = randn(5)
        c = randn(5)

        M = MulAdd(2.0,A,B,3.0,C)
        @test size(M) == size(C)
        @test size(M,1) == size(C,1)
        @test size(M,2) == size(C,2)
        @test_broken size(M,3) == size(C,3)
        @test length(M) == length(C)
        @test axes(M) == axes(C)
        @test eltype(M) == Float64
        @test materialize(M) ≈ 2.0A*B + 3.0C

        @test_throws DimensionMismatch materialize(MulAdd(2.0,A,randn(3),1.0,B))
        @test_throws DimensionMismatch materialize(MulAdd([1,2],A,B,[1,2],C))
        @test_throws DimensionMismatch materialize(MulAdd(2.0,A,B,3.0,randn(3,4)))
        @test_throws DimensionMismatch materialize(MulAdd(2.0,A,B,3.0,randn(5,5)))

        B = randn(5,5)
        C = randn(5,5)
        @test materialize(MulAdd(2.0,Diagonal(A),Diagonal(B),3.0,Diagonal(C))) == 
            muladd!(2.0,Diagonal(A),Diagonal(B),3.0,Diagonal(copy(C))) == 2.0Diagonal(A)*Diagonal(B) + 3.0*Diagonal(C)
        @test_broken materialize(MulAdd(2.0,Diagonal(A),Diagonal(B),3.0,Diagonal(C))) isa Diagonal

        @test materialize(MulAdd(1.0, Eye(5), A, 3.0, C)) == muladd!(1.0, Eye(5), A, 3.0, copy(C)) == A + 3.0C
        @test materialize(MulAdd(1.0, A, Eye(5), 3.0, C)) == muladd!(1.0, A, Eye(5), 3.0, copy(C)) == A + 3.0C

        @testset "Degenerate" begin
            C = BigFloat.(randn(3,3))
            @test muladd!(1.0, BigFloat.(randn(3,0)), randn(0,3), 2.0, copy(C) ) == 2C
            C = BigFloat.(randn(0,0))
            @test muladd!(1.0, BigFloat.(randn(0,3)), randn(3,0), 2.0, copy(C)) == C
        end

        @testset "Generic muladd" begin
            A = view(reshape(1:6,2,3),1:2,:)
            @test IndexStyle(typeof(A)) == IndexCartesian()
            b = randn(3)
            c = similar(b,2)
            @test muladd!(1.0, A, b, 0.0, c) == A*b
        end
    end

    @testset "Q" begin
        Q = qr(randn(5,5)).Q
        b = randn(5)
        B = randn(5,5)
        @test Q*b == ArrayLayouts.lmul!(Q, copy(b)) == mul(Q,b)
        @test Q*B == ArrayLayouts.lmul!(Q, copy(B)) == mul(Q,B)
        @test B*Q == ArrayLayouts.rmul!(copy(B), Q) == mul(B,Q)
        @test Q*Q ≈ mul(Q,Q)
        @test Q'*b == ArrayLayouts.lmul!(Q', copy(b)) == mul(Q',b)
        @test Q'*B == ArrayLayouts.lmul!(Q', copy(B)) == mul(Q',B)
        @test B*Q' == ArrayLayouts.rmul!(copy(B), Q') == mul(B,Q')
        @test Q*Q' ≈ mul(Q,Q')
        @test Q'*Q' ≈ mul(Q',Q')
        @test Q'*Q ≈ mul(Q',Q)
    end

    @testset "Mul" begin
        A = randn(5,5)
        b = randn(5)
        B = randn(5,5)
        
        M = Mul(A,b)
        @test size(M) == (size(M,1),) == (5,)
        @test length(M) == 5
        @test axes(M) == (axes(M,1),) == (Base.OneTo(5),)
        @test M[1] ≈ M[CartesianIndex(1)] ≈ (A*b)[1]
        @test ArrayLayouts.mul!(similar(b), A, b) ≈ A*b

        M = Mul(A,B)
        @test size(M) == (size(M,1),size(M,2)) == (5,5)
        @test length(M) == 25
        @test axes(M) == (axes(M,1),axes(M,2)) == (Base.OneTo(5),Base.OneTo(5))
        @test M[1,1] ≈ M[CartesianIndex(1,1)] ≈ M[1] ≈ (A*B)[1,1]
        @test similar(M) isa Matrix{Float64}
        @test similar(M, Int) isa Matrix{Int}
        @test similar(M, Int, (Base.OneTo(5),)) isa Vector{Int}

        M = Mul(b, b')
        @test size(M) == (size(M,1),size(M,2)) == (5,5)
        @test length(M) == 25
        @test axes(M) == (axes(M,1),axes(M,2)) == (Base.OneTo(5),Base.OneTo(5))
        @test M[1,1] ≈ M[CartesianIndex(1,1)] ≈ M[1] ≈ (b*b')[1,1]
    end

    @testset "Dot" begin
        a = randn(5)
        b = randn(5)
        @test ArrayLayouts.dot(a,b) == mul(a',b) == dot(a,b)
        @test eltype(Dot(a,1:5)) == Float64
    end
end