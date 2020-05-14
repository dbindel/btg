using LinearAlgebra

include("./kernel.jl")
include("./transform.jl")
include("./incremental.jl")
include("./kernel_system.jl")

@doc """
"""
struct BTG{G<:AbstractTransform,K<:AbstractCorrelation,C}
    horizon::Int
    n::Int
    g::G
    k::K
    comp::ComputeData
    comp_k::Vector{C}
end
function BTG()
end
