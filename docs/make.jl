using ACEcore
using Documenter

DocMeta.setdocmeta!(ACEcore, :DocTestSetup, :(using ACEcore); recursive=true)

makedocs(;
    modules=[ACEcore],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/ACEcore.jl/blob/{commit}{path}#{line}",
    sitename="ACEcore.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/ACEcore.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/ACEcore.jl",
    devbranch="main",
)
