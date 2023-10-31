using ArrayLayouts
import Aqua
using Base64
using FillArrays
using Random
using SparseArrays
using Test

import ArrayLayouts: MemoryLayout, @_layoutlmul, triangulardata

@testset "Project quality" begin
    Aqua.test_all(ArrayLayouts,
    	ambiguities = false,
    	piracy = (; broken=true),
    	project_toml_formatting = VERSION >= v"1.7",
    )
end

Random.seed!(0)

include("test_utils.jl")
include("test_layouts.jl")
include("test_muladd.jl")
include("test_ldiv.jl")
include("test_layoutarray.jl")
include("test_cumsum.jl")
