using RxGP
using Documenter

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(RxGP, :DocTestSetup, :(using RxGP); recursive=true)

makedocs(;
    modules=[RxGP],
    authors="HoangMHNguyen <m.h.n.hoang@tue.nl>, ofSingularMind <a.h.ledbetter@tue.nl>, and contributors",
    sitename="RxGP.jl",
    checkdocs = :exports,
    format=Documenter.HTML(;
        canonical="https://ReactiveBayes.github.io/RxGP.jl",
        edit_link="main",
        assets=[
            "assets/header.css",
            "assets/header.js",
            "assets/theme.css",
        ],
    ),
    pages=[
        "Home" => "index.md",
        "User guide" => [
            "Getting started" => "manuals/getting-started.md",
        ],
        "Library" => [
            "Factor nodes"            => "library/nodes.md",
            "Meta objects"            => "library/meta.md",
            "Kernel specification"    => "library/kernels.md",
            "Approximation methods"   => "library/approximation.md",
            "Prediction"              => "library/prediction.md",
            "Cache utilities"         => "library/cache.md",
            "Helper utilities"        => "library/helpers.md",
            "Type aliases"            => "library/types.md",
        ],
        "Examples" => [
            "Notebook examples" => "examples/overview.md",
            "Usage walkthrough"  => "examples/usage.md",
        ],
        "Contributing" => "contributing/guide.md",
    ],
)

deploydocs(;
    repo="github.com/ReactiveBayes/RxGP.jl",
    devbranch="main",
)
