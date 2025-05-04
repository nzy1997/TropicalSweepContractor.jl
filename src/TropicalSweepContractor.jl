module TropicalSweepContractor


using TropicalNumbers
using JuMP
using HiGHS
using GenericTensorNetworks
using LinearAlgebra
using BooleanInference

export factoring_tensornetwork,PlanarTensorNetwork

export sweep_contract!

include("sweep_contract.jl")
using .SweepContract
include("factoring.jl")
include("tropicalsvd.jl")
include("mod2.jl")
end
