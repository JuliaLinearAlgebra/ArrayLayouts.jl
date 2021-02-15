using ArrayLayouts, LinearAlgebra, FillArrays, Test
import ArrayLayouts: MemoryLayout, DenseRowMajor, DenseColumnMajor, StridedLayout,
                        ConjLayout, RowMajor, ColumnMajor, UnitStride,
                        SymmetricLayout, HermitianLayout, UpperTriangularLayout,
                        UnitUpperTriangularLayout, LowerTriangularLayout,
                        UnitLowerTriangularLayout, ScalarLayout, UnknownLayout,
                        hermitiandata, symmetricdata, FillLayout, ZerosLayout, OnesLayout,
                        DiagonalLayout, TridiagonalLayout, SymTridiagonalLayout, colsupport, rowsupport,
                        subdiag, supdiag, BidiagonalLayout, bidiagonaluplo

struct FooBar end
struct FooNumber <: Number end

@testset "MemoryLayout" begin
    @testset "Trivial" begin
        @test MemoryLayout(3.141) == MemoryLayout(Float64) == MemoryLayout(Int) == MemoryLayout(FooNumber) == ScalarLayout()
        @test MemoryLayout(FooBar()) == MemoryLayout(FooBar) == UnknownLayout()

        A = randn(6)
        @test MemoryLayout(A) == MemoryLayout(Base.ReshapedArray(A,(2,3),())) == 
            MemoryLayout(reinterpret(Float32,A)) == MemoryLayout(A) == DenseColumnMajor()
        
        @test MemoryLayout(view(A,1:3)) == DenseColumnMajor()
        @test MemoryLayout(view(A,Base.OneTo(3))) == DenseColumnMajor()
        @test MemoryLayout(view(A,:)) == DenseColumnMajor()
        @test MemoryLayout(view(A,CartesianIndex(1,1))) == DenseColumnMajor()
        @test MemoryLayout(view(A,1:2:4)) == StridedLayout()

        A = randn(6,6)
        V = view(A, 1:3,:)
        @test MemoryLayout(V) == MemoryLayout(V) == ColumnMajor()
    end

    @testset "adjoint and transpose MemoryLayout" begin
        A = [1.0 2; 3 4]
        @test MemoryLayout(A') == DenseRowMajor()
        @test MemoryLayout(transpose(A)) == DenseRowMajor()
        B = [1.0+im 2; 3 4]
        @test MemoryLayout(B') == ConjLayout{DenseRowMajor}()
        @test MemoryLayout(transpose(B)) == DenseRowMajor()
        VA = view(A, 1:1, 1:1)
        @test MemoryLayout(VA') == RowMajor()
        @test MemoryLayout(transpose(VA)) == RowMajor()
        VB = view(B, 1:1, 1:1)
        @test MemoryLayout(VB') == ConjLayout{RowMajor}()
        @test MemoryLayout(transpose(VB)) == RowMajor()
        VA = view(A, 1:2:2, 1:2:2)
        @test MemoryLayout(VA') == StridedLayout()
        @test MemoryLayout(transpose(VA)) == StridedLayout()
        VB = view(B, 1:2:2, 1:2:2)
        @test MemoryLayout(VB') == ConjLayout{StridedLayout}()
        @test MemoryLayout(transpose(VB)) == StridedLayout()
        VA2 = view(A, [1,2], :)
        @test MemoryLayout(VA2') == UnknownLayout()
        @test MemoryLayout(transpose(VA2)) == UnknownLayout()
        VB2 = view(B, [1,2], :)
        @test MemoryLayout(VB2') == UnknownLayout()
        @test MemoryLayout(transpose(VB2)) == UnknownLayout()
        VA2 = view(A, 1:2, :)
        @test MemoryLayout(VA2') == RowMajor()
        @test MemoryLayout(transpose(VA2)) == RowMajor()
        VB2 = view(B, 1:2, :)
        @test MemoryLayout(VB2') == ConjLayout{RowMajor}()
        @test MemoryLayout(transpose(VB2)) == RowMajor()
        VA2 = view(A, :, 1:2)
        @test MemoryLayout(VA2') == DenseRowMajor()
        @test MemoryLayout(transpose(VA2)) == DenseRowMajor()
        VB2 = view(B, :, 1:2)
        @test MemoryLayout(VB2') == ConjLayout{DenseRowMajor}()
        @test MemoryLayout(transpose(VB2)) == DenseRowMajor()
        VAc = view(A', 1:1, 1:1)
        @test MemoryLayout(VAc) == RowMajor()
        VAt = view(transpose(A), 1:1, 1:1)
        @test MemoryLayout(VAt) == RowMajor()
        VBc = view(B', 1:1, 1:1)
        @test MemoryLayout(VBc) == ConjLayout{RowMajor}()
        VBt = view(transpose(B), 1:1, 1:1)
        @test MemoryLayout(VBt) == RowMajor()

        @test MemoryLayout(view(randn(5)',[1,3])) == UnknownLayout()

        @testset "DualLayout" begin
            a = randn(5)
            @test MemoryLayout(a') isa DualLayout{DenseRowMajor}
            @test MemoryLayout(transpose(a)) isa DualLayout{DenseRowMajor}
            @test MemoryLayout(view(a',:,1:3)) isa DualLayout{RowMajor}
            @test MemoryLayout(view(a',1,1:3)) isa DenseColumnMajor
            @test MemoryLayout(view(a',1:1,1:3)) isa RowMajor
            @test (a')[:,1:3] == layout_getindex(a',:,1:3)
            @test (a')[1:1,1:3] == layout_getindex(a',1:1,1:3)
            @test layout_getindex(a',:,1:3) isa Adjoint
            @test layout_getindex(a',1:1,1:3) isa Array

            @test ArrayLayouts._copyto!(similar(a'), a') == a'
        end
    end

    @testset "Bi/Tridiagonal" begin
        T = Tridiagonal(randn(5),randn(6),randn(5))
        S = SymTridiagonal(T.d, T.du)
        Bl = Bidiagonal(T.d, T.dl, :L)
        Bu = Bidiagonal(T.d, T.du, :U)

        @test MemoryLayout(T) isa TridiagonalLayout
        @test MemoryLayout(Adjoint(T)) isa TridiagonalLayout
        @test MemoryLayout(Transpose(T)) isa TridiagonalLayout
        @test MemoryLayout(S) isa SymTridiagonalLayout
        @test MemoryLayout(Adjoint(S)) isa SymTridiagonalLayout
        @test MemoryLayout(Transpose(S)) isa SymTridiagonalLayout
        @test MemoryLayout(Bl) isa BidiagonalLayout
        @test MemoryLayout(Adjoint(Bl)) isa BidiagonalLayout
        @test MemoryLayout(Transpose(Bl)) isa BidiagonalLayout
        @test MemoryLayout(Bu) isa BidiagonalLayout
        @test MemoryLayout(Adjoint(Bu)) isa BidiagonalLayout
        @test MemoryLayout(Transpose(Bu)) isa BidiagonalLayout

        @test bidiagonaluplo(Bl) == bidiagonaluplo(Adjoint(Bu)) == 'L'
        @test bidiagonaluplo(Bu) == bidiagonaluplo(Adjoint(Bl)) == 'U'

        @test diag(T) == diag(T') == diag(S) == diag(Bl) == diag(Bu)
        @test supdiag(T) == subdiag(Adjoint(T)) == subdiag(Transpose(T)) == 
                    supdiag(S) == subdiag(S) == 
                    supdiag(Bu) == subdiag(Adjoint(Bu)) == subdiag(Transpose(Bu))
        @test subdiag(T) == supdiag(Adjoint(T)) == supdiag(Transpose(T)) == 
                subdiag(Bl) == supdiag(Adjoint(Bl)) == supdiag(Transpose(Bl)) ==
                T.dl

        @test colsupport(T,3) == rowsupport(T,3) == colsupport(S,3) == rowsupport(S,3) == 2:4
        @test colsupport(T,3:6) == rowsupport(T,3:6) == colsupport(S,3:6) == rowsupport(S,3:6) == 2:6
        @test colsupport(Bl,3) == rowsupport(Bu,3) == rowsupport(Adjoint(Bl),3) == 3:4
        @test rowsupport(Bl,3) == colsupport(Bu,3) == colsupport(Adjoint(Bl),3) == 2:3
        @test colsupport(Bl,3:6) == rowsupport(Bu,3:6) == 3:6
        @test colsupport(Bu,3:6) == rowsupport(Bl,3:6) == 2:6

        @test MemoryLayout(Bidiagonal(view(randn(10),[1,2,3]), view(randn(10),[1,2]), :U)) isa BidiagonalLayout{UnknownLayout}
        @test MemoryLayout(SymTridiagonal(view(randn(10),[1,2,3]), view(randn(10),[1,2]))) isa SymTridiagonalLayout{UnknownLayout}
        @test MemoryLayout(Tridiagonal(view(randn(10),[1,2]), view(randn(10),[1,2,3]), view(randn(10),[1,2]))) isa TridiagonalLayout{UnknownLayout}

        @testset "Triangular of Tridiagonal" begin
            U = UpperTriangular(T)
            L = LowerTriangular(T)
            @test MemoryLayout(U) isa BidiagonalLayout{DenseColumnMajor}
            @test MemoryLayout(L) isa BidiagonalLayout{DenseColumnMajor}
            @test diag(U) == diag(L) == diag(T)
            @test subdiag(L) == subdiag(T)
            @test supdiag(U) == supdiag(T)
            @test bidiagonaluplo(U) == 'U'
            @test bidiagonaluplo(L) == 'L'
            @test_throws MethodError subdiag(U)
        end
    end

    @testset "Symmetric/Hermitian" begin
        A = [1.0 2; 3 4]
        @test MemoryLayout(Symmetric(A)) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(Hermitian(A)) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(Transpose(Symmetric(A))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(Transpose(Hermitian(A))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(Adjoint(Symmetric(A))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(Adjoint(Hermitian(A))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(view(Symmetric(A),:,:)) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(view(Hermitian(A),:,:)) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(Symmetric(A')) == SymmetricLayout{DenseRowMajor}()
        @test MemoryLayout(Hermitian(A')) == SymmetricLayout{DenseRowMajor}()
        @test MemoryLayout(Symmetric(transpose(A))) == SymmetricLayout{DenseRowMajor}()
        @test MemoryLayout(Hermitian(transpose(A))) == SymmetricLayout{DenseRowMajor}()

        @test symmetricdata(Symmetric(A)) ≡ A
        @test symmetricdata(Hermitian(A)) ≡ A
        @test symmetricdata(Transpose(Symmetric(A))) ≡ A
        @test symmetricdata(Transpose(Hermitian(A))) ≡ A
        @test symmetricdata(Adjoint(Symmetric(A))) ≡ A
        @test symmetricdata(Adjoint(Hermitian(A))) ≡ A
        @test symmetricdata(view(Symmetric(A),:,:)) ≡ A
        @test symmetricdata(view(Hermitian(A),:,:)) ≡ A
        @test symmetricdata(Symmetric(A')) ≡ A'
        @test symmetricdata(Hermitian(A')) ≡ A'
        @test symmetricdata(Symmetric(transpose(A))) ≡ transpose(A)
        @test symmetricdata(Hermitian(transpose(A))) ≡ transpose(A)

        @test colsupport(Symmetric(A),2) ≡ colsupport(Symmetric(A),1:2) ≡ 
                rowsupport(Symmetric(A),2) ≡ rowsupport(Symmetric(A),1:2) ≡ 1:2
                @test colsupport(Hermitian(A),2) ≡ colsupport(Hermitian(A),1:2) ≡ 
                rowsupport(Hermitian(A),2) ≡ rowsupport(Hermitian(A),1:2) ≡ 1:2

        B = [1.0+im 2; 3 4]
        @test MemoryLayout(Symmetric(B)) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(Hermitian(B)) == HermitianLayout{DenseColumnMajor}()
        @test MemoryLayout(Transpose(Symmetric(B))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(Transpose(Hermitian(B))) == UnknownLayout()
        @test MemoryLayout(Adjoint(Symmetric(B))) == UnknownLayout()
        @test MemoryLayout(Adjoint(Hermitian(B))) == HermitianLayout{DenseColumnMajor}()
        @test MemoryLayout(view(Symmetric(B),:,:)) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(view(Hermitian(B),:,:)) == HermitianLayout{DenseColumnMajor}()
        @test MemoryLayout(Symmetric(B')) == UnknownLayout()
        @test MemoryLayout(Hermitian(B')) == UnknownLayout()
        @test MemoryLayout(Symmetric(transpose(B))) == SymmetricLayout{DenseRowMajor}()
        @test MemoryLayout(Hermitian(transpose(B))) == HermitianLayout{DenseRowMajor}()

        @test symmetricdata(Symmetric(B)) ≡ B
        @test hermitiandata(Hermitian(B)) ≡ B
        @test symmetricdata(Transpose(Symmetric(B))) ≡ B
        @test hermitiandata(Adjoint(Hermitian(B))) ≡ B
        @test symmetricdata(view(Symmetric(B),:,:)) ≡ B
        @test hermitiandata(view(Hermitian(B),:,:)) ≡ B
        @test symmetricdata(Symmetric(B')) ≡ B'
        @test hermitiandata(Hermitian(B')) ≡ B'
        @test symmetricdata(Symmetric(transpose(B))) ≡ transpose(B)
        @test hermitiandata(Hermitian(transpose(B))) ≡ transpose(B)

    end

    @testset "Symmetric of Banded" begin
        @eval struct BandedMock{T} <: AbstractMatrix{T} end
        ArrayLayouts.colsupport(A::BandedMock, j) = j:min(j+1, 4)
        ArrayLayouts.rowsupport(A::BandedMock, j) = max(j-1, 1):j
        ArrayLayouts.MemoryLayout(::Type{<:BandedMock}) = DenseColumnMajor()
        Base.size(::BandedMock) = (4, 4)

        A = BandedMock{Float64}()
        
        for X in (Symmetric(A), Hermitian(A))
            @test colsupport(X, 1) == rowsupport(X, 1) == 1:1
            @test colsupport(X, 2) == rowsupport(X, 2) == 2:2
            @test colsupport(X, 3) == rowsupport(X, 3) == 3:3
            @test colsupport(X, 4) == rowsupport(X, 4) == 4:4
        end
        for X in (Symmetric(A, :L), Hermitian(A, :L))
            @test colsupport(X, 1) == rowsupport(X, 1) == 1:2
            @test colsupport(X, 2) == rowsupport(X, 2) == 1:3
            @test colsupport(X, 3) == rowsupport(X, 3) == 2:4
            @test colsupport(X, 4) == rowsupport(X, 4) == 3:4
        end
    end

    @testset "Bidiagonal" begin
        B = Bidiagonal(randn(6),randn(5),:U)
        Bc = Bidiagonal(randn(6) .+ 0im,randn(5) .+ 1im,:U)
        S = Symmetric(B)
        H = Hermitian(B)
        Sc = Symmetric(Bc)
        Hc = Hermitian(Bc)

        @test MemoryLayout(S) isa SymTridiagonalLayout
        @test MemoryLayout(H) isa SymTridiagonalLayout
        @test MemoryLayout(Sc) isa SymTridiagonalLayout
        @test MemoryLayout(Hc) isa HermitianLayout

        @test diag(S) == diag(B)
        @test subdiag(S) == supdiag(S) == supdiag(B)

        @test colsupport(S,3) == colsupport(H,3) == colsupport(Sc,3) == colsupport(Hc,3) == 2:4
        @test rowsupport(S,3) == rowsupport(H,3) == rowsupport(Sc,3) == rowsupport(Hc,3) == 2:4
    end

    @testset "triangular MemoryLayout" begin
        A = [1.0 2; 3 4]
        B = [1.0+im 2; 3 4]
        for (TriType, TriLayout, TriLayoutTrans) in ((UpperTriangular, UpperTriangularLayout, LowerTriangularLayout),
                              (UnitUpperTriangular, UnitUpperTriangularLayout, UnitLowerTriangularLayout),
                              (LowerTriangular, LowerTriangularLayout, UpperTriangularLayout),
                              (UnitLowerTriangular, UnitLowerTriangularLayout, UnitUpperTriangularLayout))
            @test MemoryLayout(TriType(A)) == TriLayout{DenseColumnMajor}()
            @test MemoryLayout(TriType(transpose(A))) == TriLayout{DenseRowMajor}()
            @test MemoryLayout(TriType(A')) == TriLayout{DenseRowMajor}()
            @test MemoryLayout(transpose(TriType(A))) == TriLayoutTrans{DenseRowMajor}()
            @test MemoryLayout(TriType(A)') == TriLayoutTrans{DenseRowMajor}()

            @test MemoryLayout(TriType(B)) == TriLayout{DenseColumnMajor}()
            @test MemoryLayout(TriType(transpose(B))) == TriLayout{DenseRowMajor}()
            @test MemoryLayout(TriType(B')) == TriLayout{ConjLayout{DenseRowMajor}}()
            @test MemoryLayout(transpose(TriType(B))) == TriLayoutTrans{DenseRowMajor}()
            @test MemoryLayout(TriType(B)') == TriLayoutTrans{ConjLayout{DenseRowMajor}}()
        end

        @test MemoryLayout(UpperTriangular(B)') == MemoryLayout(LowerTriangular(B'))
    end

    @testset "Reinterpreted/Reshaped" begin
       @test MemoryLayout(reinterpret(Float32, UInt32[1 2 3 4 5])) == DenseColumnMajor()
       @test MemoryLayout(reinterpret(Float32, UInt32[1 2 3 4 5]')) == UnknownLayout()
       @test MemoryLayout(Base.__reshape((randn(6),IndexLinear()),(2,3))) == DenseColumnMajor()
       @test MemoryLayout(Base.__reshape((1:6,IndexLinear()),(2,3))) == UnknownLayout()
    end

    @testset "Fill" begin
        @test MemoryLayout(Fill(1,10)) == FillLayout()
        @test MemoryLayout(Ones(10)) == OnesLayout()
        @test MemoryLayout(Zeros(10)) == ZerosLayout()
        @test MemoryLayout(SubArray(Fill(1,10),(1:3,))) == FillLayout()
        @test MemoryLayout(view(Fill(1,10),1:3,1)) == FillLayout()
        @test MemoryLayout(SubArray(Fill(1,10),([1,3,2],))) == FillLayout()
        @test MemoryLayout(reshape(Fill(1,10),2,5)) == FillLayout()
        @test MemoryLayout(Fill(1+0im,10)') == DualLayout{FillLayout}()
        @test MemoryLayout(Adjoint(Fill(1+0im,10,2))) == FillLayout()
        @test MemoryLayout(Transpose(Fill(1+0im,10,2))) == FillLayout()

        @test layout_getindex(Fill(1,10), 1:3) ≡ Fill(1,3)
        @test layout_getindex(Ones{Int}(1,10), 1, 1:3) ≡ Ones{Int}(3)
        @test layout_getindex(Zeros{Int}(5,10,12), 1, 1:3,4:6) ≡ Zeros{Int}(3,3)

        # views of Fill no longer create Sub Arrays, but are supported
        # as there was no strong need to delete their support
        v = SubArray(Fill(1,10),(1:3,))
        @test ArrayLayouts.sub_materialize(v) ≡ Fill(1,3)
        @test ArrayLayouts._copyto!(Vector{Float64}(undef,3), v) == ones(3)
    end

    @testset "Triangular col/rowsupport" begin
        A = randn(5,5)
        @test colsupport(UpperTriangular(A),3) ≡ Base.OneTo(3)
        @test rowsupport(UpperTriangular(A),3) ≡ 3:5
        @test colsupport(LowerTriangular(A),3) ≡ 3:5
        @test rowsupport(LowerTriangular(A),3) ≡ Base.OneTo(3)
    end

    @testset "PermutedDimsArray" begin
        A = [1.0 2; 3 4]
        @test MemoryLayout(PermutedDimsArray(A, (1,2))) == DenseColumnMajor()
        @test MemoryLayout(PermutedDimsArray(A, (2,1))) == DenseRowMajor()
        @test MemoryLayout(transpose(PermutedDimsArray(A, (2,1)))) == DenseColumnMajor()
        @test MemoryLayout(adjoint(PermutedDimsArray(A, (2,1)))) == DenseColumnMajor()
        B = [1.0+im 2; 3 4]
        @test MemoryLayout(PermutedDimsArray(B, (2,1))) == DenseRowMajor()
        @test MemoryLayout(transpose(PermutedDimsArray(B, (2,1)))) == DenseColumnMajor()
        @test MemoryLayout(adjoint(PermutedDimsArray(B, (2,1)))) == ConjLayout{DenseColumnMajor}()

        C = view(ones(10,20,30), 2:9, 3:18, 4:27);
        @test MemoryLayout(C) == ColumnMajor()
        @test MemoryLayout(PermutedDimsArray(C, (1,2,3))) == ColumnMajor()
        @test MemoryLayout(PermutedDimsArray(C, (1,3,2))) == UnitStride{1}()

        @test MemoryLayout(PermutedDimsArray(C, (3,1,2))) == UnitStride{2}()
        @test MemoryLayout(PermutedDimsArray(C, (2,1,3))) == UnitStride{2}()

        @test MemoryLayout(PermutedDimsArray(C, (3,2,1))) == RowMajor()
        @test MemoryLayout(PermutedDimsArray(C, (2,3,1))) == UnitStride{3}()

        revC = PermutedDimsArray(C, (3,2,1));
        @test MemoryLayout(PermutedDimsArray(revC, (3,2,1))) == ColumnMajor()
        @test MemoryLayout(PermutedDimsArray(revC, (3,1,2))) == UnitStride{1}()

        D = ones(10,20,30,40);
        @test MemoryLayout(D) == DenseColumnMajor()
        @test MemoryLayout(PermutedDimsArray(D, (1,2,3,4))) == DenseColumnMajor()
        @test MemoryLayout(PermutedDimsArray(D, (1,4,3,2))) == UnitStride{1}()

        @test MemoryLayout(PermutedDimsArray(D, (4,1,3,2))) == UnitStride{2}()
        @test MemoryLayout(PermutedDimsArray(D, (2,1,4,3))) == UnitStride{2}()

        @test MemoryLayout(PermutedDimsArray(D, (4,3,2,1))) == DenseRowMajor()
        @test MemoryLayout(PermutedDimsArray(D, (4,2,1,3))) == UnitStride{3}()

        twoD = PermutedDimsArray(D, (3,1,2,4));
        MemoryLayout(PermutedDimsArray(twoD, (2,1,4,3))) == UnitStride{1}()

        revD = PermutedDimsArray(D, (4,3,2,1));
        MemoryLayout(PermutedDimsArray(revD, (4,3,2,1))) == DenseColumnMajor()
        MemoryLayout(PermutedDimsArray(revD, (4,2,3,1))) == UnitStride{1}()


        issorted((1,2,3,4))
        # Fails on Julia 1.4, in tests. Could use BenchmarkTools.@ballocated instead.
        @test_skip 0 == @allocated issorted((1,2,3,4))
        reverse((1,2,3,4))
        @test_skip 0 == @allocated reverse((1,2,3,4))
        MemoryLayout(revD)
        @test 0 == @allocated MemoryLayout(revD)
    end
end
