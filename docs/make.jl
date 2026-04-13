using RxGP
using Documenter

DocMeta.setdocmeta!(RxGP, :DocTestSetup, :(using RxGP); recursive=true)

function convert_notebooks()
    examples_dir = joinpath(@__DIR__, "..", "examples")
    output_dir   = joinpath(@__DIR__, "src")

    mkpath(output_dir)

    notebooks = filter(f -> endswith(f, ".ipynb"), readdir(examples_dir))

    for nb in notebooks
        nb_path = joinpath(examples_dir, nb)
        println("nbconvert → ", nb)

        run(`jupyter nbconvert --to markdown $(nb_path) --output-dir $(output_dir)`)
    end
end

convert_notebooks()

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
            "Examples" => ["GPRegression.md","GPSSM.md"],
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