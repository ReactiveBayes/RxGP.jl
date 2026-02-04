using RxGP
using Documenter

DocMeta.setdocmeta!(RxGP, :DocTestSetup, :(using RxGP); recursive=true)

makedocs(;
    modules=[RxGP],
    clean=true,
    sitename="RxGP.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ReactiveBayes.github.io/RxGP.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "How to use" => [
            "Getting Started" => "getting_started.md",
            "API Reference" => "api_reference.md",
        ],  
    ],
)


deploydocs(;
    repo="github.com/ReactiveBayes/RxGP.jl",
    devbranch="main",
)

if !haskey(ENV, "CI")
    index = joinpath(@__DIR__, "build", "index.html")
    if Sys.isapple()
        run(`open $index`)
    elseif Sys.iswindows()
        run(`cmd /c start "" $index`)
    else
        run(`xdg-open $index`)
    end
end