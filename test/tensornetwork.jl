using Test
using TropicalSweepContractor
using TropicalSweepContractor.SweepContractor
using TropicalSweepContractor: and_tensor, true_tensor, false_tensor, or_tensor, xor_tensor, one_tensor,multiplier_tensor,factoring_tensornetworks

@testset "basic tensors" begin
    # C = A and B
    LTN = LabelledTensorNetwork{Char,Float64}()
    LTN['D'] = Tensor(['A','B','C'],and_tensor(Float64),1.0,2.0)
    LTN['A'] = Tensor(['D'], true_tensor(Float64), 0.0, 0.0)
    LTN['B'] = Tensor(['D'], false_tensor(Float64), 1.0, 0.0)
    LTN['C'] = Tensor(['D'], false_tensor(Float64), 0.0, 1.0)

    sweep = sweep_contract(LTN,2,4)
    @test ldexp(sweep...) == 1.0

    # C = A or B
    LTN = LabelledTensorNetwork{Char,Float64}()
    LTN['D'] = Tensor(['A','B','C'],or_tensor(Float64),1.0,2.0)
    LTN['A'] = Tensor(['D'], true_tensor(Float64), 0.0, 0.0)
    LTN['B'] = Tensor(['D'], false_tensor(Float64), 1.0, 0.0)
    LTN['C'] = Tensor(['D'], true_tensor(Float64), 0.0, 1.0)

    sweep = sweep_contract(LTN,2,4)
    @test ldexp(sweep...) == 1.0

    # C = A xor B
    LTN = LabelledTensorNetwork{Char,Float64}()
    LTN['D'] = Tensor(['A','B','C'],xor_tensor(Float64),1.0,2.0)
    LTN['A'] = Tensor(['D'], true_tensor(Float64), 0.0, 0.0)
    LTN['B'] = Tensor(['D'], true_tensor(Float64), 1.0, 0.0)
    LTN['C'] = Tensor(['D'], false_tensor(Float64), 0.0, 1.0)

    sweep = sweep_contract(LTN,2,4)
    @test ldexp(sweep...) == 1.0
end

@testset "multiplier tensor" begin
    mat = multiplier_tensor(Float64)
    LTN = LabelledTensorNetwork{String,Float64}()
    LTN["A"] = Tensor(["pi","po","qi","qo","ci","co","si","so"],mat,1.0,2.0)
    LTN["pi"] = Tensor(["A"],false_tensor(Float64),0.0,0.0)
    LTN["po"] = Tensor(["A"],false_tensor(Float64),1.0,0.0)
    LTN["qi"] = Tensor(["A"],true_tensor(Float64),2.0,0.0)
    LTN["qo"] = Tensor(["A"],true_tensor(Float64),3.0,0.0)
    LTN["ci"] = Tensor(["A"],true_tensor(Float64),4.0,0.0)
    LTN["co"] = Tensor(["A"],true_tensor(Float64),5.0,0.0)
    LTN["si"] = Tensor(["A"],true_tensor(Float64),6.0,0.0)
    LTN["so"] = Tensor(["A"],false_tensor(Float64),7.0,0.0)

    sweep = sweep_contract(LTN,2,4)
    @test ldexp(sweep...) == 1.0

    mat = multiplier_tensor(Float64)
    LTN = LabelledTensorNetwork{String,Float64}()
    LTN["A"] = Tensor(["pi","po","qi","qo","ci","co","si","so"],mat,1.0,2.0)
    LTN["pi"] = Tensor(["A"],false_tensor(Float64),0.0,0.0)
    LTN["po"] = Tensor(["A"],false_tensor(Float64),1.0,0.0)
    LTN["qi"] = Tensor(["A"],true_tensor(Float64),2.0,0.0)
    LTN["qo"] = Tensor(["A"],false_tensor(Float64),3.0,0.0)
    LTN["ci"] = Tensor(["A"],true_tensor(Float64),4.0,0.0)
    LTN["co"] = Tensor(["A"],true_tensor(Float64),5.0,0.0)
    LTN["si"] = Tensor(["A"],true_tensor(Float64),6.0,0.0)
    LTN["so"] = Tensor(["A"],false_tensor(Float64),7.0,0.0)

    sweep = sweep_contract(LTN,2,4)
    @test ldexp(sweep...) == 0.0
end

@testset "factoring_tensornetwork" begin
    factoring_tensornetwork(5,5,10)
end