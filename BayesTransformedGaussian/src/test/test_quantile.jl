using LinearAlgebra
using Test
using Distributions
using DataFrames
using CSV
using Polynomials

include("quadrature/quadrature.jl")
include("transforms/transforms.jl")
include("priors/priors.jl")
include("bayesopt/incremental.jl")
include("kernels/kernel.jl")
include("dataStructs.jl")
include("computation/buffers0.jl") #datastruct, kernel, incremental, quadrature
include("model0.jl") #buffers, datastructs, several auxiliary
include("computation/tdist.jl") #model0 and buffer0


df = DataFrame(CSV.File("datasets/abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age
normalizing_constant = maximum(target)
target = target/normalizing_constant #normalization

const ind = 120:125
const x = data[ind, 1:3] 
#choose a subset of variables to be regressors for the mean
const Fx = data[ind, 1:2] 
const y = float(target[ind])
const pind = 10:10 #prediction index
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
#prediction data
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
Fx0 = reshape(data[pind, 1:2], 1, 2)
x0 = reshape(data[pind, 1:3], 1, d) 
rangeθ = [100.0 200]
rangeλ = [0.5 5] #we will always used 1 range scale for lambda

btg1 = btg(trainingData1, rangeθ, rangeλ)
(dpdf, pdf, cdf, quant0) = solve(btg1)
dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
pdf_fixed = y0 -> pdf(x0, Fx0, y0)
cdf_fixed = y0 -> cdf(x0, Fx0, y0)
quant0_fixed = p -> quant0(x0, Fx0, p) 
