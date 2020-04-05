using ArrayLayouts, LinearAlgebra, FillArrays, Test
import ArrayLayouts: MemoryLayout, DenseRowMajor, DenseColumnMajor, StridedLayout,
                        ConjLayout, RowMajor, ColumnMajor, UnknownLayout,
                        SymmetricLayout, HermitianLayout, UpperTriangularLayout,
                        UnitUpperTriangularLayout, LowerTriangularLayout,
                        UnitLowerTriangularLayout, ScalarLayout,
                        hermitiandata, symmetricdata, FillLayout, ZerosLayout,
                        DiagonalLayout, colsupport, rowsupport

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
    end

    @testset "Symmetric/Hermitian MemoryLayout" begin
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
        @test MemoryLayout(Ones(10)) == FillLayout()
        @test MemoryLayout(Zeros(10)) == ZerosLayout()
        @test MemoryLayout(view(Fill(1,10),1:3)) == FillLayout()
        @test MemoryLayout(view(Fill(1,10),1:3,1)) == FillLayout()
    end

    @testset "Triangular col/rowsupport" begin
        A = randn(5,5)
        @test colsupport(UpperTriangular(A),3) ≡ Base.OneTo(3)
        @test rowsupport(UpperTriangular(A),3) ≡ 3:5
        @test colsupport(LowerTriangular(A),3) ≡ 3:5
        @test rowsupport(LowerTriangular(A),3) ≡ Base.OneTo(3)
    end
end
