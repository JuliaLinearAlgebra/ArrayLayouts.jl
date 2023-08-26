using Documenter
using ArrayLayouts

makedocs(
    sitename = "ArrayLayouts",
    modules = [ArrayLayouts]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/JuliaLinearAlgebra/ArrayLayouts.jl.git",
)
