using LinearAlgebra
using Distributions
using SpecialFunctions
using PDMats
using CSV
#using Profile
#using ProfileView
using TimerOutputs
include("kernel.jl")
include("quadrature.jl")
include("btgCompute.jl")
include("examples.jl")
include("transforms.jl")


"""
Define inference problem by supplying
    *s: observation locations
    *X: covariates for observed locations
    *z: observed values/labels
    *X0: matrix of covariates at prediction location
    *s0: prediction location
"""
mutable struct setting{T<:Array{Float64, 2}, S<:Array{Float64, 1}}
    s::T
    s0::T
    X::T
    X0::T
    z::S
end



"""
Marginalize out theta and lambda to obtain Bayesian predictive density p(z0|z)

Arg \"type\" refers to type of quadrature rule used to integrate out θ, options are 
    1) Gaussian (does not use derivatives)
    2) Turan (uses higher derivatives)
Note: Gaussian quadrature is always used to integrate out λ
"""
function getBtgDensity(setting::setting{Array{Float64, 2}, Array{Float64, 1}}, rangeθ::Array{Float64, 1}, rangeλ::Array{Float64, 1}, type = "Gaussian")
    if type == "Turan"
        nodesWeightsθ = getTuranQuadratureData() #use 12 Gauss-Turan integration nodes and weights by default
    else if type == "Gaussian"
        nodesWeightsθ = getGaussQuadraturedata()
    end
    affineTransformNodes(nodesWeightsθ, rangeθ)
    #always use Gauss quadrature to marginalize out λ
    nodesWeightsλ = getGaussQuadraturedata()
    affineTransformNodes(nodesWeightsλ, rangeλ)
    weightsGrid = getWeightsGrid(setting, nodesWeightsθ, nodesWeightsλ, boxCoxObj)
    evalTGrid = createTensorGrid(setting, meshθ::Array{Float64, 1}, meshλ::Array{Float64, 1}, nodesWeightsθ, nodesWeightsλ, type = "Gaussian")
    function density(z0)
        dot(evalTGrid(z0), weightsGrid)
    end
    return density
end
