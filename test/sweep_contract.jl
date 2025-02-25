using Test
using TropicalSweepContractor
using TropicalSweepContractor:splitMPStensor

@testset "splitMPStensor" begin
    x = rand(Int,2,3,4,5,6)
    splitMPStensor(x,[10,20,30])
end