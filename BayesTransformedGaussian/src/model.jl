using LinearAlgebra
using Distributions
using SpecialFunctions
using PDMats
using CSV
#using Profile
#using ProfileView
using TimerOutputs
include("kernels/kernel.jl")
include("computation/derivatives.jl")
include("transforms.jl")
include("transforms.jl")
include("settings.jl")
include("quadrature/tensorgrid.jl")
include("priors.jl")

"""
Obtain Bayesian predictive density function p(z0|z) and cumulative density function P(z0|z)
by marginalizing out theta and lambda.

Arg \"quadtype\" refers to type of quadrature rule used to integrate out θ, options are 
    1) Gaussian (does not use derivatives)
    2) Turan (uses higher derivatives)

Note: Gaussian quadrature is always used to integrate out λ
"""
function getBtgDensity(train::trainingData{A, B}, test::testingData{A}, rangeθ::B, rangeλ::B, transforms, quadtype = "Gaussian", priortype = "Uniform") where A<:Array{Float64, 2} where B<:Array{Float64, 1}
    if quadtype == "Turan"
        nodesWeightsθ = getTuranQuadratureData() #use 12 Gauss-Turan integration nodes and weights by default
    elseif quadtype == "Gaussian"
        nodesWeightsθ = getGaussQuadraturedata()
    else
        throw(ArgumentError("Quadrature rule not recognized"))
    end

    nodesWeightsθ = getGaussQuadraturedata()
    affineTransformNodes(nodesWeightsθ, rangeθ)
    #always use Gauss quadrature to marginalize out λ
    nodesWeightsλ = getGaussQuadraturedata()
    affineTransformNodes(nodesWeightsλ, rangeλ)
    priorθ = initialize_prior(rangeθ, priortype); priorλ = initialize_prior(rangeλ, priortype); 
    (pdf, cdf) = getTensorGrid(train, test, priorθ, priorλ, nodesWeightsθ, nodesWeightsλ, transforms, quadtype)
end
