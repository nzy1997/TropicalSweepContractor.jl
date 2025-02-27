using Test
using TropicalSweepContractor
using TropicalSweepContractor: multiplier_tensor
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
    ft = factoring_tensornetwork(2,2,10;T = TropicalNumbers.TropicalAndOr)
    @test sweep_contract!(ft,40,40) == false

    ft = factoring_tensornetwork(2,2,17;T = TropicalNumbers.TropicalAndOr)
    @test sweep_contract!(ft,40,40) == true
end

