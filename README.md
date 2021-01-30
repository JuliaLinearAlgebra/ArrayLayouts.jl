# ArrayLayouts.jl
A Julia package for describing array layouts and more general fast linear algebra

[![Build Status](https://github.com/JuliaMatrices/ArrayLayouts.jl/workflows/CI/badge.svg)](https://github.com/JuliaMatrices/ArrayLayouts.jl/actions)
[![codecov](https://codecov.io/gh/JuliaMatrices/ArrayLayouts.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaMatrices/ArrayLayouts.jl)

This package implements a trait-based framework for describing array layouts such as column major, row major, etc. that can be dispatched 
to appropriate BLAS or optimised Julia linear algebra routines. This supports a much wider class of matrix types than Julia's in-built `StridedArray`. Here is an example:
```julia
julia> using ArrayLayouts

julia> A = randn(10_000,10_000); x = randn(10_000); y = similar(x);

julia> V = view(Symmetric(A),:,:)';

julia> @time mul!(y, A, x); # Julia does not recognize that V is symmetric
  0.040255 seconds (4 allocations: 160 bytes)

julia> @time muladd!(1.0, V, x, 0.0, y); # ArrayLayouts does and is 3x faster as it calls BLAS
  0.017677 seconds (4 allocations: 160 bytes)
```

## Internal design

The package is based on assigning a `MemoryLayout` to every array, which is used for dispatch. For example, 
```julia
julia> MemoryLayout(A) # Each column of A is column major, and columns stored in order
DenseColumnMajor()

julia> MemoryLayout(view(A, 1:3,:))  # Each column of A is column major
ColumnMajor()

julia> MemoryLayout(V) # A symmetric version, whose storage is DenseColumnMajor
SymmetricLayout{DenseColumnMajor}()
```

This is then used by `muladd!(α, A, B, β, C)`, `ArrayLayouts.lmul!(A, B)`, and `ArrayLayouts.rmul!(A, B)` to
lower to the correct BLAS calls via lazy objects `MulAdd(α, A, B, β, C)`, `Lmul(A, B)`, `Rmul(A, B)` which are materialized,
in analogy to `Base.Broadcasted`. 

Note there is also a higher level function `mul(A, B)` that materializes via `Mul(A, B)`, which uses the layout of `A` and `B`
to further reduce to either `MulAdd`, `Lmul`, and `Rmul`. 