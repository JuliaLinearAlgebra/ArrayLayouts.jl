using Documenter
using ArrayLayouts

makedocs(
    sitename = "ArrayLayouts",
    format = Documenter.HTML(),
    modules = [ArrayLayouts]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
