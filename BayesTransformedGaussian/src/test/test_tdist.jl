using Test
using Distributions
using DataFrames
using CSV
using Polynomials

include("../kernels/kernel.jl")
include("../computation/finitedifference.jl")
include("../computation/tdist.jl")
include("../transforms/transforms.jl")
include("../datastructs.jl")

df = DataFrame(CSV.File("../datasets/abalone.csv"))
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

d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
#rangeθ = [2.0 5; 4 7; 5 10]  #number of length scales is height of rangeθ
rangeθ = [100 200]
rangeλ = [0.5 5] #we will always used 1 range scale for lambda
btg1 = btg(trainingData1, rangeθ, rangeλ)
(dpdf, pdf, cdf) = solve(btg1)
Fx0 = reshape(data[pind, 1:2], 1, 2)
x0 = reshape(data[pind, 1:3], 1, d)   
dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
pdf_fixed = y0 -> pdf(x0, Fx0, y0)
cdf_fixed = y0 -> cdf(x0, Fx0, y0)
println("Checking derivative of btg_pdf...")
(r1, r2, plt1, pol) = checkDerivative(pdf_fixed, cdf_fixed, 0.5, nothing, 8, 16, 10)
display(plt1)
println(pol)
println("Plotting dpdf, pdf, and cdf...")
#plt(dpdf_fixed, .1, 1, 100)
plt(pdf_fixed, .1, 1, 100)
plt!(cdf_fixed, .1, 1, 100, title = "PDF and CDF of Bayesian Predictive Distribution")

