module ArrayLayoutsTests

import ArrayLayouts
import Aqua
import Random
using Test

@testset "Project quality" begin
    Aqua.test_all(ArrayLayouts,
    	ambiguities = false,
    	piracies = (; broken=true),
    )
end

Random.seed!(0)

include("test_utils.jl")
include("test_layouts.jl")
include("test_muladd.jl")
include("test_ldiv.jl")
include("test_layoutarray.jl")
include("test_cumsum.jl")

end
