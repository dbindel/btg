"""
Define inference problem by supplying
    s: observation locations
    X: covariates for observed locations
    z: observed values/labels
    X0: matrix of covariates at prediction location
    s0: prediction location
"""
struct setting{T<:Array{Float64, 2}, S<:Array{Float64, 1}}
    s::T
    s0::T
    X::T
    X0::T
    z::S
end