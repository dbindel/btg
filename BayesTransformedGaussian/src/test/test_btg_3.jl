#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(The BTG Program)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################################################################
#copy this box into a file to use the BTG program                              
using LinearAlgebra                                                           
using Test                                                                                    
using Distributions                                                           
using DataFrames                                                                    
using CSV                                                                         
using Polynomials      
using PyPlot
include("../validation/det_loocv.jl")                                             
include("../validation/loocv.jl")                                             
include("../quadrature/quadrature.jl")                                             
include("../transforms/transforms.jl")                                             
include("../priors/priors.jl")                                                     
include("../bayesopt/incremental.jl")                                              
include("../kernels/kernel.jl")                                                    
include("../datastructs.jl")                                                       
include("../computation/buffers0.jl") #datastruct, kernel, incremental, quadrature 
include("../model0.jl") #buffers, datastructs, several auxiliary   
include("../computation/finitedifference.jl")                                                                
include("../computation/tdist.jl") #model0 and buffer0                             
#################################################################################


df = DataFrame(CSV.File("../datasets/abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age
normalizing_constant = maximum(target)
target = target/normalizing_constant #normalization

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~(btg1)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Attributes: 
# - single length scale
# - 2-dimensional covariates
# - 3-multidimensional location vectors

#ind = 120:125
#ind = [1:9;11:25]
#ind = 1:30
ind = 100:150
#ind = 45:65
posx = 1:4 #
posc = 1:1
x = data[ind, posx] 
#choose a subset of variables to be regressors for the mean

Fx = data[ind, posc] 
y = float(target[ind])
#pind = 10:10 #prediction index
pind = 27:27
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
#prediction data
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
Fx0 = reshape(data[pind, posc], 1, length(posc))
x0 = reshape(data[pind, posx], 1, length(posx)) 
rangeθ = [10.0 2000]
#rangeθ = [50.0 1000]
rangeλ = [-1.0 1] #we will always used 1 range scale for lambda
btg1 = btg(trainingData1, rangeθ, rangeλ; quadtype = ["MonteCarlo", "MonteCarlo"])
#θ1 = btg1.nodesWeightsθ.nodes[1, 6] #pick some theta value which doubles as quadrature node
#λ1 = btg1.nodesWeightsλ.nodes[3]
##################################################################################################

println("Test point attributes")
println("Location:")
println(data[pind, posx])
println("Covariates")
println(data[pind, posc])
println("Label")
println(target[pind])

#(_, _, _) = solve(btg1)  
(pdf, cdf, dpdf) = solve(btg1);  
a = y0 -> dpdf(x0, Fx0, y0); 
b = y0 -> pdf(x0, Fx0, y0);
c = y0 -> cdf(x0, Fx0, y0);

#println("Checking derivative of btg_pdf...")
#(_, _, plt1, pol1) = checkDerivative(b, a, 0.5, nothing, 4, 12, 10); #first arg is function, second arg is derivative
#(_, _, plt2, pol2) = checkDerivative(c, b, 0.5, nothing, 4, 12, 10); #first arg is function, second arg is derivative

#PyPlot.figure(1)
#(x, y) = plt_data(b, .0001, 2, 100)
#(x1, y1) = plt_data(c, .0001, 2, 100)

#PyPlot.plot(x, y)
#PyPlot.plot(x1, y1)
#plt(b, .01, 1; label = "pdf")
#plt!(c, .01, 1; label = "cdf")
#plt(target[pind])

#for i = 151:160
for i = 220:230
    pind = i:i
    #prediction data
    Fx0 = reshape(data[pind, posc], 1, length(posc))
    x0 = reshape(data[pind, posx], 1, length(posx))
    b = y0 -> pdf(x0, Fx0, y0);
    c = y0 -> cdf(x0, Fx0, y0);
    figure(i)
    (x, y) = plt_data(b, .01, 2, 400)
    (x1, y1) = plt_data(c, .01, 2, 400)
    PyPlot.plot(x, y)
    PyPlot.plot(x1, y1)
    PyPlot.axvline(x=target[i])
end


