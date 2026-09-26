using Documenter
using ForsetiMultivariate

DocMeta.setdocmeta!(ForsetiMultivariate, :DocTestSetup, :(using ForsetiMultivariate); recursive = true)

makedocs(;
    sitename = "ForsetiMultivariate.jl",
    modules = [ForsetiMultivariate],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = ["Home" => "index.md"],
)

deploydocs(;
    repo = "github.com/Forseti-jl/ForsetiMultivariate.jl.git",
    devbranch = "main",
)
