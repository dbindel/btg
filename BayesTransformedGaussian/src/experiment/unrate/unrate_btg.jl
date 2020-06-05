using Dates
using ArgParse
using Printf
using Random
using GaussianProcesses

# before_all = Dates.now()

include("../../btg.jl")
path = "../../datasets/"
include(path * "load_unrate.jl")
include(path * "load_unrate_full.jl")

data, label = unrate_full(path)


#data, label = unrate(path)
data, label = unrate_full(path)
max_label = maximum(label)
label = label ./ max_label

xs = copy(data)
xs = xs/maximum(data)

###
###  Manually enter parsed args entries so we can run this file as a script
###
##
parsed_args = Dict()
push!(parsed_args, "randseed" => 1234)
push!(parsed_args, "lmax" => 0.3)
push!(parsed_args, "lmin" => 0.2)
push!(parsed_args, "percent_train" => 0.4)
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

# training set
p_train = parsed_args["percent_train"]
n_train = (Int64)(round(p_train * length(data)))
n_test = length(data) - n_train
"""
Input percent train
"""
function sample(inds, percent)
    num = (Int64)(round(percent * length(data)))
    #train = sort(rand(1:490, (num,)))
    ind_train = sort(randperm(length(data))[1:num])
    ind_test = copy(inds)
    deleteat!(ind_test, ind_train)
    #work with indices up until this point
    x_train = xs[ind_train]
    x_test = xs[ind_test]
    return ind_train, x_train, ind_test, x_test 
end

ind_train, x_train, ind_test, x_test  = sample(data, p_train) #training and testing pts
train_label = label[ind_train] 
test_label = label[ind_test]
y_train = train_label

x = reshape(x_train, n_train, 1)
Fx = covariate_fun(x, parsed_args["p"])

trainingData0 = trainingData(x, Fx, train_label) 

d = getDimension(trainingData0); n = getNumPts(trainingData0); p = getCovDimension(trainingData0)

x_test = reshape(x_test, n_test, 1)
Fx_test = covariate_fun(x_test, parsed_args["p"])
y_test = test_label

elapsedmin = 0

#parameter setting
# myquadtype = parsed_args["sparse"] ? ["SparseCarlo", "SparseCarlo"] : ["QuasiMonteCarlo", "QuasiMonteCarlo"]
myquadtype = ["Gaussian", "Gaussian"]
#rangeλ = [-3.0 3] 
#rangeλ = reshape([-0.2], 1, 1)
#rangeλ = reshape([-0.2 0.2], 1, 2) #compare to this
#rangeλ = reshape([-0.2], 1, 1) #BETTER BY 0.5 PERCENT
#rangeλ = reshape([-0.001], 1, 1) #compare above to this

rangeλ = reshape([-5.0 5.0], 1, 2) #BETTER BY 0.5 PERCENT
lmin = parsed_args["lmin"]
lmax = parsed_args["lmax"]
#rangeθ = [1/lmax^2 1/lmin^2]
#rangeθ = reshape([50.0 150.0], 1, 2)
rangeθ = reshape([100.0], 1, 1)
@info "rangeθ, rangel:" rangeθ, [lmin lmax]
# rangeθ = [0.111 25]   
# build btg model

function load_unrate_btg()
    return btg(trainingData0, rangeθ, rangeλ; priorθ = inverseUniform(rangeθ), quadtype = myquadtype)
end

function getseed()
    return randseed
end
