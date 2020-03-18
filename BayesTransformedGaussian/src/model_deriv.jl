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
Marginalize out theta and lambda to obtain Bayesian predictive density
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
