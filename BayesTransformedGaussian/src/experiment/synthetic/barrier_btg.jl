using Dates
using ArgParse
using Printf
using Random
using GaussianProcesses

# before_all = Dates.now()

include("../../tools/split.jl")
include("../../btg.jl")
include("../../covariatefun.jl")

###
###  Manually enter parsed args entries so we can run this file as a script
###
##
parsed_args = Dict()
push!(parsed_args, "randseed" => 1234)
push!(parsed_args, "lmax" => 0.5)
push!(parsed_args, "lmin" => 0.2)
#push!(parsed_args, "noiselevel" => 1)
push!(parsed_args, "N" => 100)
push!(parsed_args, "p_train" => 0.5)
push!(parsed_args, "p" => 1) #dimension of covariates
push!(parsed_args, "N_plot"=>200) #number pts plotted on graph
push!(parsed_args, "upper_bound_estimate" => 10) #crude estimate for upper bound of function for box-constrained opt
randseed = parsed_args["randseed"]; rng = MersenneTwister(randseed)
push!(parsed_args, "log_scale" => false)

Random.seed!(randseed);

#fun(x) = .5sin(10x) + 1+ 1/((x-5)/4)^2
fun(x) = sin(10x) .+ 2 .+ 0.5*rand(1)[1] #noisy settin
# training set
N = parsed_args["N"]
xs = range(0.001, stop = 4.9999, length = N)
#xs = vcat(range(.001, stop=4.8, length=N), range(4.81, stop = 4.99, length = Int64(round(N/4))))
#xs = vcat(xs, range(4.991, stop = 4.9999, length = Int64(round(N/8))))
inds = collect(1:1:length(xs))
ind_train, x_train, ind_test, x_test  = sample(inds, parsed_args["p_train"])
x_test = reshape(x_test, length(x_test), 1) #reshape
x_train = reshape(x_train, length(x_train), 1) #reshape

n_train = length(ind_train)
n_test = length(ind_test)
ys = fun.(xs)

#data = sort((rand(Float64, (n_train, )) .- 0.5)*2*pi)
y_train = ys[ind_train]
Fx = covariate_fun(x_train, parsed_args["p"])
trainingData0 = trainingData(reshape(x_train, length(x_train), 1), Fx, y_train) 
d = getDimension(trainingData0); n = getNumPts(trainingData0); p = getCovDimension(trainingData0)

Fx_test = covariate_fun(reshape(x_test, length(x_test), 1), parsed_args["p"])
y_test = ys[ind_test]
elapsedmin = 0

#parameter setting
# myquadtype = parsed_args["sparse"] ? ["SparseCarlo", "SparseCarlo"] : ["QuasiMonteCarlo", "QuasiMonteCarlo"]
myquadtype = ["Gaussian", "Gaussian"]
rangeλ = [0.5 1.5] 
#rangeλ = reshape([1.0], 1, 1)
lmin = parsed_args["lmin"]
lmax = parsed_args["lmax"]
#rangeθ = [1/lmax^2 1/lmin^2]
#rangeθ = reshape([140.0], 1, 1)
#rangeθ = reshape([145.0], 1, 1)
rangeθ = reshape([14.0], 1, 1)
@info "rangeθ, rangel:" rangeθ, [lmin lmax]
# rangeθ = [0.111 25]   
# build btg model

function load_synthetic_btg()
    return btg(trainingData0, rangeθ, rangeλ; priorθ = inverseUniform(rangeθ), quadtype = myquadtype)
end

function getseed()
    return randseed
end