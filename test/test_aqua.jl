module AquaTests

import Aqua
import ArrayLayouts
using Test

@testset "Project quality" begin
    Aqua.test_all(ArrayLayouts,
    	ambiguities = false,
    	piracies = (; broken=true),
    )
end

end
