
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

############################### Load in data
df = DataFrame(CSV.File("datasets/abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age
normalizing_constant = maximum(target)
target = target/normalizing_constant #normalization

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(Problem Setup)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ind = 11:200
posx = 1:7 #locations 
posc = 1:1 #covariates 
x = data[ind, posx] 
#choose a subset of variables to be regressors for the mean
Fx = data[ind, posc]
y = float(target[ind])
pind = 10:10 #prediction index
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
#prediction data
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
Fx0 = reshape(data[pind, posc], 1, length(posc)) #Pind associate with test data
x0 = reshape(data[pind, posx], 1, length(posx))
#rangeθ = [10.0 20]
#rangeθ = [10.0 1000.0; 10.0 1000]
rangeθ = [1000.0 10000.0]
rangeλ = [-3 3.0] #we will always used 1 range scale for lambda
#################################################################################
#init btg object for testing various functions
btg1 = btg(trainingData1, rangeθ, rangeλ)
   
θ1 = btg1.nodesWeightsθ.nodes[1, 6] #pick some theta value which doubles as quadrature node

if true #test solve_btg
    #output = @capture_err begin
        (dpdf, pdf, cdf) = solve(btg1)
    #end;
    dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
    pdf_fixed = y0 -> pdf(x0, Fx0, y0)
    cdf_fixed = y0 -> cdf(x0, Fx0, y0)
    a = dpdf_fixed
    b = pdf_fixed
    c = cdf_fixed

    #(r1, r2, plt1, pol) = checkDerivative(c, b, 0.5, nothing, 1, 17, 20) #function first, derivative second
    #display(plt1)
    #println(pol)
    #display(plt1)
    if true
    #plt(a, .01, 1, 150)
    println("Plotting pdf...")
    (x, y) = plt(b, .001, 1, 200, label = "pdf")
    println("Plotting cdf...")
    (x1, y1) = plt!(c, 0.001, 1, 200, label = "cdf")
    Plots.plot!(target[pind], seriestype = :vline, title = "PDF and CDF of Bayesian Predictive Distribution", label = "actual")
    end
end
