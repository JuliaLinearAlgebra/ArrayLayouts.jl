# ArrayLayouts.jl
A Julia package for describing array layouts and more general fast linear algebra

[![Travis](https://travis-ci.org/JuliaMatrices/ArrayLayouts.jl.svg?branch=master)](https://travis-ci.org/JuliaMatrices/ArrayLayouts.jl)
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