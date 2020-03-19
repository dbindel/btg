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
include("Derivatives.jl")
include("TDist.jl")
include("examples.jl")
include("transforms.jl")
include("settings.jl")
include("tensorgrid.jl")
include("priors.jl")

"""
Obtain Bayesian predictive density function p(z0|z) and cumulative density function P(z0|z)
by marginalizing out theta and lambda.

Arg \"quadtype\" refers to type of quadrature rule used to integrate out θ, options are 
    1) Gaussian (does not use derivatives)
    2) Turan (uses higher derivatives)

Note: Gaussian quadrature is always used to integrate out λ
"""
function getBtgDensity(setting::setting{Array{Float64, 2}, Array{Float64, 1}}, rangeθ::Array{Float64, 1}, rangeλ::Array{Float64, 1}, quadtype = "Gaussian", priortype = "Uniform")
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
    #weightsGrid = getWeightsGrid(setting, priorθ, priorλ, nodesWeightsθ, nodesWeightsλ, boxCoxObj)
    (pdf, cdf) = getTensorGrid(setting, priorθ, priorλ, nodesWeightsθ, nodesWeightsλ, boxCoxObj, quadtype)
end


##EVERYTHING BELOW IS DEPRECATED...
function model_deriv(example::setting{Array{Float64, 2}, Array{Float64, 1}}, pθ, dpθ, dpθ2, pλ, rangeθ, rangeλ)
    #z0 ->  Gauss_Turan(θ -> (theta_params = func(θ, example); int1D( λ -> ((g, dg, d2g) = define_fs(θ, λ, theta_params, example); [g(z0), dg(z0), d2g(z0)]), rangeλ)), rangeθ)
    function nested(theta_params::θ_params{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}, θ::Float64, z0)
        function intlambda(z0::Array{Float64, 1}, λ::Float64)
            (g, dg, d2g) = define_fs(θ, λ, theta_params, example)
            return [g(z0), dg(z0), d2g(z0)]
        end
        return int1D(λ -> intlambda(z0, float(λ)), rangeλ)
    end
    function density(z0::Array{Float64, 1})
        function int(θ::Float64)
            theta_params = func(θ, example)
            return nested(theta_params, θ, z0)
        end
        return Gauss_Turan(int, rangeθ)
    end
    return density
end

function test_model_deriv(nn=25)
    θ=2.2
    rangeλ =[1 2]
    example = getExample(1, nn, 1, 1, 2)
    println("time to compute theta-dependent params")
    println(@elapsed theta_params = func(θ, example))
    z0 = [5.0]
    λ = 4.0
    println("time to define g, dg, d2g")
    #@profview (g, dg, d2g) = define_fs(θ, λ, theta_params, example)
    @time begin
    (g, dg, d2g) = define_fs(θ, λ, theta_params, example)
    end
    println("time to compute evaluate integrand and its derivatives")
    println(@elapsed (g(1);dg(1);d2g(1)))
    #y = z0 -> int1D( λ -> ((g, dg, d2g) = define_fs(θ, λ, theta_params); 
    #[g(z0), dg(z0), d2g(z0)]), rangeλ)
end
