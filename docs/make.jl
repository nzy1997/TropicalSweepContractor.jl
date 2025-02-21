using TropicalSweepContractor
using Documenter

DocMeta.setdocmeta!(TropicalSweepContractor, :DocTestSetup, :(using TropicalSweepContractor); recursive=true)

makedocs(;
    modules=[TropicalSweepContractor],
    authors="nzy1997",
    sitename="TropicalSweepContractor.jl",
    format=Documenter.HTML(;
        canonical="https://nzy1997.github.io/TropicalSweepContractor.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nzy1997/TropicalSweepContractor.jl",
    devbranch="main",
)
