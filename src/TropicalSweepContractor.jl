module TropicalSweepContractor

using OMEinsum
using LinearAlgebra
using TropicalNumbers
using JuMP
using HiGHS

export factoring_tensornetwork,PlanarTensorNetwork

export sweep_contract!

const TensorOrder = Base.By(λ->(λ.y,λ.x))

include("tensornetwork.jl")
include("sweep_contract.jl")
include("tropicalsvd.jl")
end
