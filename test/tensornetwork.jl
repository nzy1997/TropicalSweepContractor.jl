using Test
using TropicalSweepContractor
using TropicalSweepContractor: multiplier_tensor,PlanarTensorNetwork,PlanarTensor
using TropicalNumbers
@testset "multiplier tensor" begin
    mat = multiplier_tensor(Float64)
    tag = true
    for p_i in [1,2],p_o in [1,2],q_i in [1,2],q_o in [1,2],c_i in [1,2],c_o in [1,2],s_i in [1,2],s_o in [1,2]
        tag2 = (2*(c_o-1)+s_o-1 == (p_i-1)*(q_i-1)+c_i+s_i-2)
        tag2 = tag2 && (p_i == p_o) && (q_i == q_o)
        tag = tag && (mat[p_i,p_o,q_i,q_o,c_i,c_o,s_i,s_o] == 1.0) == tag2
    end
    @test tag
end

@testset "factoring_tensornetwork" begin
    ft = factoring_tensornetwork(2,2,10)
    @show sweep_contract!(ft)

    ft = factoring_tensornetwork(2,2,2;T = TropicalNumbers.Tropical{Float64})
    @show sweep_contract!(ft)

    ft = factoring_tensornetwork(3,3,4)
    @show sweep_contract!(ft)
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

    @test brute ≈ sweep_contract!(ptn) atol=1e-10
end
