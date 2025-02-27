using Test
using TropicalSweepContractor
using TropicalSweepContractor:splitMPStensor,MPS

@testset "splitMPStensor" begin
    x = rand(Int,2,3,4,5,6)
    mps = splitMPStensor(x)
    @test length(mps) == 3
    @test mps isa MPS{Int}
end