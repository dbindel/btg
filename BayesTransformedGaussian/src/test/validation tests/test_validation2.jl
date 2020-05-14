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
#ind = [5:9;11:17]
#ind = 5:17
ind = 1:25
posx = 1:3 #
posc = 1:3
x = data[ind, posx] 
#choose a subset of variables to be regressors for the mean

Fx = data[ind, posc] 
y = float(target[ind])
pind = 10:10 #prediction index
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
#prediction data

d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
Fx0 = reshape(data[pind, posc], 1, length(posc))
x0 = reshape(data[pind, posx], 1, length(posx)) 
rangeθ = [100.0 200]
rangeλ = [0.5 3] #we will always used 1 range scale for lambda
btg1 = btg(trainingData1, rangeθ, rangeλ)
#θ1 = btg1.nodesWeightsθ.nodes[1, 6] #pick some theta value which doubles as quadrature node
#λ1 = btg1.nodesWeightsλ.nodes[3]
##################################################################################################

i=10
println("Validation point attributes")
println("Location:")
println(getPosition(btg1.trainingData)[i, :])
println("Covariates")
println(getCovariates(btg1.trainingData)[i, :])
println("Label")
println(getLabel(btg1.trainingData)[i])

function validate(btg1::btg, i)
    (pdf1, cdf1, dpdf1) = solve(btg1; validate = i)  
    a1 = y0 -> dpdf1(x0, Fx0, y0) 
    b1 = y0 -> pdf1(x0, Fx0, y0)
    c1 = y0 -> cdf1(x0, Fx0, y0)
    return (a1, b1, c1)
end

(a2, b2, c2) = validate(btg1, i);

#println("Checking derivative of btg_pdf...")
(_, _, plt1, pol1) = checkDerivative(b2, a2, 0.5, nothing, 4, 12, 10); #first arg is function, second arg is derivative
(_, _, plt2, pol2) = checkDerivative(c2, b2, 0.5, nothing, 4, 12, 10); #first arg is function, second arg is derivative

(x, y) = plt_data(b2, .1, 1)
(x1, y1) = plt_data(c2, .1, 1)

PyPlot.figure(10)
PyPlot.plot(x, y)
#PyPlot.plot(x1, y1)

#plt(b, .01, 1; label = "pdf")
#plt!(c, .01, 1; label = "cdf")
#plt(target[pind])

#(pdf, cdf, dpdf) = solve(btg1)  
#a1 = y0 -> dpdf(x0, Fx0, y0) 
#b1 = y0 -> pdf(x0, Fx0, y0)
#c1 = y0 -> cdf(x0, Fx0, y0)

#i=4

#plt!(b2, .01, 1)
#plt!(c2, .01, 1)
#plt!(target[ind[i]])

keys(btg1.θλbuffer_dict)|> y-> filter(x -> x[1][1] >130 && x[1][1]<132, y)