
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(The BTG Program)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################################################################
#copy this box into a file to use the BTG program                               #
using LinearAlgebra                                                             #
using Test                                                                      #                  
using Distributions                                                             #
using DataFrames                                                                #          
using CSV                                                                       #       
using Polynomials                                                               #
                                                                                # 
include("computation/finitedifference.jl")                                      #             
include("quadrature/quadrature.jl")                                             #
include("transforms/transforms.jl")                                             #
include("priors/priors.jl")                                                     #
include("bayesopt/incremental.jl")                                              #
include("kernels/kernel.jl")                                                    #
include("datastructs.jl")                                                       #
include("computation/buffers0.jl") #datastruct, kernel, incremental, quadrature #
include("model0.jl") #buffers, datastructs, several auxiliary                   #
include("computation/tdist.jl") #model0 and buffer0                             #
#################################################################################


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

#init btg object for testing various functions
btg1 = btg(trainingData1, rangeθ, rangeλ)
    θ1 = btg1.nodesWeightsθ.nodes[1, 6] #pick some theta value which doubles as quadrature node
    println(θ1) 

if false #test comp_tdist 
    (dpdf, pdf, cdf) = comp_tdist(btg1, [θ1], [1.4]) 
    dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
    pdf_fixed = y0 -> pdf(x0, Fx0, y0)
    cdf_fixed = y0 -> cdf(x0, Fx0, y0)
    a = pdf_fixed
    b = cdf_fixed
    c = dpdf_fixed

    rr = 8:16
    h = abs.(b.(.5 .+ [2.718 ^(-i) for i=rr]) .- b.(.5) - [2.718 ^(-i) for i = rr] .* a(.5))
    xx = [2.718 ^(-i) for i=rr]
    lxx = log.(xx)
    lh = log.(h)
    polyfit(lxx, lh, 1)

    (r1, r2, plt1, pol) = checkDerivative(a, c, 0.5, nothing, 8, 16, 10) #

    @testset "comp_tdist" begin
        @test coeffs(pol)[end] ≈ 2 atol = 1e-1
    end

end

if true #test solve_btg
    (dpdf, pdf, cdf, _) = solve(btg1)
    dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
    pdf_fixed = y0 -> pdf(x0, Fx0, y0)
    cdf_fixed = y0 -> cdf(x0, Fx0, y0)
    a = dpdf_fixed
    b = pdf_fixed
    c = cdf_fixed

    (r1, r2, plt1, pol) = checkDerivative(c, b, 0.5, nothing, 1, 17, 20) #function first, derivative second
    display(plt1)
    println(pol)
    display(plt1)
    #plt(a, .01, 1, 150)
    #plt!(b, .01, 1, 150, title = "PDF and CDF of Bayesian Predictive Distribution")
    #plt!(c, 0.01, 1, 150)
end
