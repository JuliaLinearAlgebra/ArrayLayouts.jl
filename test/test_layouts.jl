using ArrayLayouts, LinearAlgebra, FillArrays, Test
import ArrayLayouts: MemoryLayout, DenseRowMajor, DenseColumnMajor, StridedLayout,
                        ConjLayout, RowMajor, ColumnMajor, UnitStride,
                        SymmetricLayout, HermitianLayout, UpperTriangularLayout,
                        UnitUpperTriangularLayout, LowerTriangularLayout,
                        UnitLowerTriangularLayout, ScalarLayout, UnknownLayout,
                        hermitiandata, symmetricdata, FillLayout, ZerosLayout,
                        DiagonalLayout, colsupport, rowsupport

struct FooBar end
struct FooNumber <: Number end

@testset "MemoryLayout" begin
    @testset "Trivial" begin
        @test MemoryLayout(3.141) == MemoryLayout(Float64) == MemoryLayout(Int) == MemoryLayout(FooNumber) == ScalarLayout()
        @test MemoryLayout(FooBar()) == MemoryLayout(FooBar) == UnknownLayout()

        A = randn(6)
        @test MemoryLayout(typeof(A)) == MemoryLayout(typeof(Base.ReshapedArray(A,(2,3),()))) ==
            MemoryLayout(typeof(reinterpret(Float32,A))) == MemoryLayout(A) == DenseColumnMajor()

        @test MemoryLayout(typeof(view(A,1:3))) == DenseColumnMajor()
        @test MemoryLayout(typeof(view(A,Base.OneTo(3)))) == DenseColumnMajor()
        @test MemoryLayout(typeof(view(A,:))) == DenseColumnMajor()
        @test MemoryLayout(typeof(view(A,CartesianIndex(1,1)))) == DenseColumnMajor()
        @test MemoryLayout(typeof(view(A,1:2:4))) == StridedLayout()

        A = randn(6,6)
        V = view(A, 1:3,:)
        @test MemoryLayout(typeof(V)) == MemoryLayout(V) == ColumnMajor()
    end

    @testset "adjoint and transpose MemoryLayout" begin
        A = [1.0 2; 3 4]
        @test MemoryLayout(typeof(A')) == DenseRowMajor()
        @test MemoryLayout(typeof(transpose(A))) == DenseRowMajor()
        B = [1.0+im 2; 3 4]
        @test MemoryLayout(typeof(B')) == ConjLayout{DenseRowMajor}()
        @test MemoryLayout(typeof(transpose(B))) == DenseRowMajor()
        VA = view(A, 1:1, 1:1)
        @test MemoryLayout(typeof(VA')) == RowMajor()
        @test MemoryLayout(typeof(transpose(VA))) == RowMajor()
        VB = view(B, 1:1, 1:1)
        @test MemoryLayout(typeof(VB')) == ConjLayout{RowMajor}()
        @test MemoryLayout(typeof(transpose(VB))) == RowMajor()
        VA = view(A, 1:2:2, 1:2:2)
        @test MemoryLayout(typeof(VA')) == StridedLayout()
        @test MemoryLayout(typeof(transpose(VA))) == StridedLayout()
        VB = view(B, 1:2:2, 1:2:2)
        @test MemoryLayout(typeof(VB')) == ConjLayout{StridedLayout}()
        @test MemoryLayout(typeof(transpose(VB))) == StridedLayout()
        VA2 = view(A, [1,2], :)
        @test MemoryLayout(typeof(VA2')) == UnknownLayout()
        @test MemoryLayout(typeof(transpose(VA2))) == UnknownLayout()
        VB2 = view(B, [1,2], :)
        @test MemoryLayout(typeof(VB2')) == UnknownLayout()
        @test MemoryLayout(typeof(transpose(VB2))) == UnknownLayout()
        VA2 = view(A, 1:2, :)
        @test MemoryLayout(typeof(VA2')) == RowMajor()
        @test MemoryLayout(typeof(transpose(VA2))) == RowMajor()
        VB2 = view(B, 1:2, :)
        @test MemoryLayout(typeof(VB2')) == ConjLayout{RowMajor}()
        @test MemoryLayout(typeof(transpose(VB2))) == RowMajor()
        VA2 = view(A, :, 1:2)
        @test MemoryLayout(typeof(VA2')) == DenseRowMajor()
        @test MemoryLayout(typeof(transpose(VA2))) == DenseRowMajor()
        VB2 = view(B, :, 1:2)
        @test MemoryLayout(typeof(VB2')) == ConjLayout{DenseRowMajor}()
        @test MemoryLayout(typeof(transpose(VB2))) == DenseRowMajor()
        VAc = view(A', 1:1, 1:1)
        @test MemoryLayout(typeof(VAc)) == RowMajor()
        VAt = view(transpose(A), 1:1, 1:1)
        @test MemoryLayout(typeof(VAt)) == RowMajor()
        VBc = view(B', 1:1, 1:1)
        @test MemoryLayout(typeof(VBc)) == ConjLayout{RowMajor}()
        VBt = view(transpose(B), 1:1, 1:1)
        @test MemoryLayout(typeof(VBt)) == RowMajor()

        @test MemoryLayout(typeof(view(randn(5)',[1,3]))) == UnknownLayout()
    end

    @testset "Symmetric/Hermitian MemoryLayout" begin
        A = [1.0 2; 3 4]
        @test MemoryLayout(typeof(Symmetric(A))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Hermitian(A))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Transpose(Symmetric(A)))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Transpose(Hermitian(A)))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Adjoint(Symmetric(A)))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Adjoint(Hermitian(A)))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(view(Symmetric(A),:,:))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(view(Hermitian(A),:,:))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Symmetric(A'))) == SymmetricLayout{DenseRowMajor}()
        @test MemoryLayout(typeof(Hermitian(A'))) == SymmetricLayout{DenseRowMajor}()
        @test MemoryLayout(typeof(Symmetric(transpose(A)))) == SymmetricLayout{DenseRowMajor}()
        @test MemoryLayout(typeof(Hermitian(transpose(A)))) == SymmetricLayout{DenseRowMajor}()

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

        B = [1.0+im 2; 3 4]
        @test MemoryLayout(typeof(Symmetric(B))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Hermitian(B))) == HermitianLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Transpose(Symmetric(B)))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Transpose(Hermitian(B)))) == UnknownLayout()
        @test MemoryLayout(typeof(Adjoint(Symmetric(B)))) == UnknownLayout()
        @test MemoryLayout(typeof(Adjoint(Hermitian(B)))) == HermitianLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(view(Symmetric(B),:,:))) == SymmetricLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(view(Hermitian(B),:,:))) == HermitianLayout{DenseColumnMajor}()
        @test MemoryLayout(typeof(Symmetric(B'))) == UnknownLayout()
        @test MemoryLayout(typeof(Hermitian(B'))) == UnknownLayout()
        @test MemoryLayout(typeof(Symmetric(transpose(B)))) == SymmetricLayout{DenseRowMajor}()
        @test MemoryLayout(typeof(Hermitian(transpose(B)))) == HermitianLayout{DenseRowMajor}()

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

    @testset "triangular MemoryLayout" begin
        A = [1.0 2; 3 4]
        B = [1.0+im 2; 3 4]
        for (TriType, TriLayout, TriLayoutTrans) in ((UpperTriangular, UpperTriangularLayout, LowerTriangularLayout),
                              (UnitUpperTriangular, UnitUpperTriangularLayout, UnitLowerTriangularLayout),
                              (LowerTriangular, LowerTriangularLayout, UpperTriangularLayout),
                              (UnitLowerTriangular, UnitLowerTriangularLayout, UnitUpperTriangularLayout))
            @test MemoryLayout(typeof(TriType(A))) == TriLayout{DenseColumnMajor}()
            @test MemoryLayout(typeof(TriType(transpose(A)))) == TriLayout{DenseRowMajor}()
            @test MemoryLayout(typeof(TriType(A'))) == TriLayout{DenseRowMajor}()
            @test MemoryLayout(typeof(transpose(TriType(A)))) == TriLayoutTrans{DenseRowMajor}()
            @test MemoryLayout(typeof(TriType(A)')) == TriLayoutTrans{DenseRowMajor}()

            @test MemoryLayout(typeof(TriType(B))) == TriLayout{DenseColumnMajor}()
            @test MemoryLayout(typeof(TriType(transpose(B)))) == TriLayout{DenseRowMajor}()
            @test MemoryLayout(typeof(TriType(B'))) == TriLayout{ConjLayout{DenseRowMajor}}()
            @test MemoryLayout(typeof(transpose(TriType(B)))) == TriLayoutTrans{DenseRowMajor}()
            @test MemoryLayout(typeof(TriType(B)')) == TriLayoutTrans{ConjLayout{DenseRowMajor}}()
        end

        @test MemoryLayout(typeof(UpperTriangular(B)')) == MemoryLayout(typeof(LowerTriangular(B')))
    end

    @testset "Reinterpreted/Reshaped" begin
       @test MemoryLayout(typeof(reinterpret(Float32, UInt32[1 2 3 4 5]))) == DenseColumnMajor()
       @test MemoryLayout(typeof(reinterpret(Float32, UInt32[1 2 3 4 5]'))) == UnknownLayout()
       @test MemoryLayout(typeof(Base.__reshape((randn(6),IndexLinear()),(2,3)))) == DenseColumnMajor()
       @test MemoryLayout(typeof(Base.__reshape((1:6,IndexLinear()),(2,3)))) == UnknownLayout()
    end

    @testset "Fill" begin
        @test MemoryLayout(typeof(Fill(1,10))) == FillLayout()
        @test MemoryLayout(typeof(Ones(10))) == FillLayout()
        @test MemoryLayout(typeof(Zeros(10))) == ZerosLayout()
        @test MemoryLayout(typeof(view(Fill(1,10),1:3))) == FillLayout()
        @test MemoryLayout(typeof(view(Fill(1,10),1:3,1))) == FillLayout()
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

        D = ones(10,20,30,40);
        @test MemoryLayout(D) == DenseColumnMajor()
        @test MemoryLayout(PermutedDimsArray(D, (1,2,3,4))) == DenseColumnMajor()
        @test MemoryLayout(PermutedDimsArray(D, (1,4,3,2))) == UnitStride{1}()

        @test MemoryLayout(PermutedDimsArray(D, (4,1,3,2))) == UnitStride{2}()
        @test MemoryLayout(PermutedDimsArray(D, (2,1,4,3))) == UnitStride{2}()

        @test MemoryLayout(PermutedDimsArray(D, (4,3,2,1))) == DenseRowMajor()
        @test MemoryLayout(PermutedDimsArray(D, (4,2,1,3))) == UnitStride{3}()

        issorted((1,2,3,4))
        @test_skip 0 == @allocated issorted((1,2,3,4)) # fails on Julia 1.4, in tests
        reverse((1,2,3,4))
        @test_skip 0 == @allocated reverse((1,2,3,4))
        revD = PermutedDimsArray(D, (4,3,2,1));
        MemoryLayout(revD)
        @test 0 == @allocated MemoryLayout(revD)
    end
end
