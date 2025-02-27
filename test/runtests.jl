using TropicalSweepContractor
using Test

@testset "factoring.jl" begin
    include("factoring.jl")
end

@testset "tropicalsvd.jl" begin
    include("tropicalsvd.jl")
end

@testset "sweep_contract.jl" begin
    include("sweep_contract.jl")
end
