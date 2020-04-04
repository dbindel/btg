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
#prediction data
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
Fx0 = reshape(data[pind, 1:2], 1, 2)
x0 = reshape(data[pind, 1:3], 1, d) 
rangeθ = [100.0 200]
rangeλ = [0.5 5] #we will always used 1 range scale for lambda

@testset "BoxCox" begin
    bc = BoxCox()
    g = (x, λ) -> bc(x, λ)
    dg = (x, λ) -> partialx(bc, x, λ)
    dg2 = (x, λ) -> partialx(bc, x, λ)
    g_fixed = x-> g(x, 1.5)
    dg_fixed = x-> dg(x, 1.5) 
    dg2_fixed = x-> dg2(x, 1.5) 
    (_, _, _, pol) = checkDerivative(dg_fixed, dg2_fixed, 0.5, nothing, 8, 16, 10)
    @test coeffs(pol)[end] ≈ 2 atol = 1e-1
    (_, _, _, pol) = checkDerivative(g_fixed, dg_fixed, 0.5, nothing, 8, 16, 10)
    @test coeffs(pol)[end] ≈ 2 atol = 1e-1
    end

if true #test comp_tdist 
    btg1 = btg(trainingData1, rangeθ, rangeλ)
    (dpdf, pdf, cdf) = comp_tdist(btg1, [.3], [1.4]) 
    dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
    pdf_fixed = y0 -> pdf(x0, Fx0, y0)
    cdf_fixed = y0 -> cdf(x0, Fx0, y0)
    
    (r1, r2, plt1, pol) = checkDerivative(pdf_fixed, cdf_fixed, 0.5, nothing, 8, 16, 10)
    display(plt1)
    plt(pdf_fixed, .01, 1, 150)
    plt!(cdf_fixed, .01, 1, 150, title = "PDF and CDF of Bayesian Predictive Distribution")
end

if false
#rangeθ = [2.0 5; 4 7; 5 10]  #number of length scales is height of rangeθ

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
end

