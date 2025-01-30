using RxGP
using Documenter

DocMeta.setdocmeta!(RxGP, :DocTestSetup, :(using RxGP); recursive=true)

makedocs(;
    modules=[RxGP],
    authors="HoangMHNguyen <m.h.n.hoang@tue.nl> and contributors",
    sitename="RxGP.jl",
    format=Documenter.HTML(;
        canonical="https://ReactiveBayes.github.io/RxGP.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ReactiveBayes/RxGP.jl",
    devbranch="main",
)
