using Dates
using ArgParse
using Printf
using Random
using GaussianProcesses

# before_all = Dates.now()

include("../../btg.jl")

###
###  Manually enter parsed args entries so we can run this file as a script
###
##
parsed_args = Dict()
push!(parsed_args, "randseed" => 1234)
push!(parsed_args, "lmax" => 0.5)
push!(parsed_args, "lmin" => 0.2)
push!(parsed_args, "noiselevel" => 1)
push!(parsed_args, "ntest" => 200)
push!(parsed_args, "ntrain" => 80)
push!(parsed_args, "p" => 1)

randseed = parsed_args["randseed"]; rng = MersenneTwister(randseed)
Random.seed!(randseed);

function covariate_fun(x, p)
    n = size(x, 1)
    d = size(x, 2)
    if p == 1
        return ones(n, 1)
    elseif p == 1 + d
        return hcat(ones(n), x)
    else
        throw(ArgumentError("Only support constant or linear convariate."))
    end
end

true_fun(x, s) = sin(x) + s*randn() + 3
true_fun_noise(x) = max(true_fun(x, 1e-1), 0)^1/3
true_fun_0(x) = true_fun(x, 0.)^1/3

# training set
n_train = parsed_args["ntrain"]
data = range(-pi, stop=pi, length=n_train)
#data = sort((rand(Float64, (n_train, )) .- 0.5)*2*pi)
target = (sin.(data) .+ 0.1 * parsed_args["noiselevel"].* randn(rng, n_train) .+ 3).^(1/3)
x = reshape(data, n_train, 1)
Fx = covariate_fun(x, parsed_args["p"])
max_train = maximum(target)
y = target ./ max_train
trainingData0 = trainingData(x, Fx, y) 
d = getDimension(trainingData0); n = getNumPts(trainingData0); p = getCovDimension(trainingData0)
# testing set 
n_test = parsed_args["ntest"]
x_test = range(-pi, stop=pi, length=n_test)
#x_test = (rand(Float64, (n_test, )) .- 0.5)*2*pi
x_test = sort(reshape(x_test, n_test, 1), dims=1)
Fx_test = covariate_fun(x_test, parsed_args["p"])
y_test_true = (sin.(x_test) .+ 3).^(1/3)
elapsedmin = 0

#parameter setting
# myquadtype = parsed_args["sparse"] ? ["SparseCarlo", "SparseCarlo"] : ["QuasiMonteCarlo", "QuasiMonteCarlo"]
myquadtype = ["Gaussian", "Gaussian"]
rangeλ = [-20.0 20.0] 
#rangeλ = reshape([0.1], 1, 1)
lmin = parsed_args["lmin"]
lmax = parsed_args["lmax"]
#rangeθ = [1/lmax^2 1/lmin^2]
rangeθ = reshape([1.0], 1, 1)
@info "rangeθ, rangel:" rangeθ, [lmin lmax]
# rangeθ = [0.111 25]   
# build btg model

function load_synthetic_btg()
    return btg(trainingData0, rangeθ, rangeλ; priorθ = inverseUniform(rangeθ), quadtype = myquadtype)
end

function getseed()
    return randseed
end