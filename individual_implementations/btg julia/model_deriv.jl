using LinearAlgebra
using Distributions
using SpecialFunctions
using PDMats
using CSV
#using Profile
#using ProfileView
using TimerOutputs
include("kernel.jl")
include("integration.jl")
include("btgDerivatives.jl")
include("examples.jl")
include("integration.jl")

"""
Define prediction/inference problem by supplying known parameters, including design matrices, 
observation locations, prediction locations, nonlinear transform, prior marginals on theta and lambda,
and anticipated ranges for theta and lambda. Returns Bayesian predictive density function, 
namely the integral of p(z_0| theta, lambda, z)*p(theta, lambda | z) over Theta x Lambda
(See Equations 8 and 12)

INPUTS: 
X: k x p design matrix 
X0: n x p design matrix
s: observed locations
s0: prediction locations
g: nonlinear transform parametrized by lambda. Usage: g(lambda, z)
gprime: derivative of g. Usage: gprime(z, lambda)
ptheta: prior marginal on theta; function handle 
plambda: prior marginal on lambda; function handle
theta: structural parameter
lambda: transform parameter
z: observed random vector
z0: unobserved random vector

OUTPUTS:
probability density function z0 -> f(z_0|z) 
"""
function define_fs(θ::Float64, λ::Float64, theta_params::θ_params{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}, example::setting{Array{Float64, 2}})
    #time = @elapsed begin
    pθ = x->1 
    dpθ = x->0
    dpθ2 = x->0
    pλ = x->1
    (main, dmain, d2main) = partial_theta(float(θ), float(λ), example, theta_params)
    (main1, dmain1, d2main1) = posterior_theta(float(θ), float(λ), pθ, dpθ, dpθ2, pλ, example, theta_params)
    f = z0 -> (main(z0)*main1); df = z0 -> (main(z0)*dmain1 .+ dmain(z0)*main1); d2f = z0 -> (d2main(z0)*main1 .+ main(z0)*d2main1 .+ 2*dmain(z0)*dmain1)
    #end
    #println("define_fs time: %s\n", time)
    return (f, df, d2f)
end

"""
This function returns a function of z0 which evaluates the integrand 
along with its higher derivatives (currently first and second derivatives)
at prescribed meshgrid locations
"""
function createTensorGrid(example::setting{Array{Float64, 2}, Array{Float64, 1}}, meshθ::Array{Float64, 1}, meshλ::Array{Float64, 1})
    l1 = length(meshθ); l2 = length(meshλ)
    function func_fixed(θ::Float64)
        return func(θ, example)
    end
    theta_param_list = Array{θ_params{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}}(undef, l1)
    for i=1:l1
        theta_param_list[i] = func_fixed(meshθ[i])
    end
    function evalTgrid(z0)
        tgrid = zeros(l1, l2, 3) #tensor grid, layers are for higher derivatives
        for i = 1:l1
            for j = 1:l2 
                (f, df, d2f) = define_fs(meshθ[i], meshλ[j], theta_param_list[i], example)
                tgrid[i, j, 1] = f(z0)
                tgrid[i, j, 2] = df(z0)
                tgrid[i, j, 3] = d2f(z0)
            end
        end
        return tgrid
    end
    return evalTgrid
end

"""
Combine Gauss Quadrature (used on the lambdas) and Turan Quadrature (used on thetas)
to marginalize out theta and lambda to obtain Bayesian predictive density
"""
function getBtgDensity(example::setting{Array{Float64, 2}, Array{Float64, 1}}, rangeθ::Array{Float64, 1}, rangeλ::Array{Float64, 1})
    bθ = rangeθ[2]; aθ = rangeθ[1]; bλ = rangeλ[2]; aλ = rangeλ[1]
    nodes_T, weights_T = getTuranData() #use 12 Gauss-Turan integration nodes and weights by default
    nodes_T = (bθ-aθ)/2 .* nodes_T .+ (bθ+aθ)/2
    nodes_G, weights_G = getGaussQuadraturedata()
    nodes_G = (bλ-aλ)/2 .* nodes_G .+ (bλ+aλ)/2 #transform nodes to span range
    n1 = length(nodes_T); m1 = size(weights_T, 2); n2 = length(nodes_G)
    weightsTensorGrid = zeros(n1, n2, m1) #tensor grid of weights
    for i = 1:n1
        for j = 1:n2
            for k = 1:m1
                weightsTensorGrid[i, j, k] = weights_T[i, k]*weights_G[j]
            end
        end
    end
    evalTGrid = createTensorGrid(example, nodes_T, nodes_G)
    function density(z0)
        dot(evalTGrid(z0), weightsTensorGrid)
    end
    return density
end

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
