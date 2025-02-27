using Test
using TropicalSweepContractor
using TropicalSweepContractor.SweepContract:splitMPStensor,MPS,PlanarTensorNetwork,PlanarTensor

@testset "splitMPStensor" begin
    x = rand(Int,2,3,4,5,6)
    mps = splitMPStensor(x)
    @test length(mps) == 3
    @test mps isa MPS{Int}
end

@testset "ABCD" begin
    using Random
    Random.seed!(1234)
    tensorA = PlanarTensor(rand(3,3,2),[1,5,6],0.0,1.0)
    tensorB = PlanarTensor(rand(3,3,3),[1,2,4],0.0,0.0)
    tensorC = PlanarTensor(rand(3,3),[2,3],1.0,0.0)
    tensorD = PlanarTensor(rand(3,3,3,2),[3,4,5,6],1.0,1.0)
    ptn = PlanarTensorNetwork([
        tensorA,
        tensorB,
        tensorC,
        tensorD
    ],6)
    brute = 0.0
    for e1=1:3, e2=1:3, e3=1:3, e4=1:3, e5=1:3, e6=1:2
        brute += tensorA.tensor[e1,e5,e6]*tensorB.tensor[e1,e2,e4]*tensorC.tensor[e2,e3]*tensorD.tensor[e3,e4,e5,e6]
    end

    @test brute ≈ sweep_contract!(ptn,3,3) atol=1e-10
end
