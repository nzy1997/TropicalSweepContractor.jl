module TropicalSweepContractor

using OMEinsum
using LinearAlgebra

export factoring_tensornetwork,PlanarTensorNetwork

export sweep_contract!

const TensorOrder = Base.By(λ->(λ.y,λ.x))

include("tensornetwork.jl")
include("sweep_contract.jl")
end
