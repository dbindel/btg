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
#ind = 50:150
#ind = [199;200; 510;530; 700; 780; 1000;3000;4010;4020;4030]
ind = 350:389
#ind = 45:65
posx = 1:7 #
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
#rangeθ = [100.0 200]
rangeθ = [10.0 2000]
rangeλ = [-1.0 1] #we will always used 1 range scale for lambda
btg1 = btg(trainingData1, rangeθ, rangeλ) #quadtype = ["SparseGrid", "MonteCarlo"])
#θ1 = btg1.nodesWeightsθ.nodes[1, 6] #pick some theta value which doubles as quadrature node
#λ1 = btg1.nodesWeightsλ.nodes[3]
##################################################################################################
PyPlot.close("all") #close existing windows


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
(_, _, plt1, pol1) = checkDerivative(b, a, 0.5, nothing, 4, 12, 10); #first arg is function, second arg is derivative
(_, _, plt2, pol2) = checkDerivative(c, b, 0.5, nothing, 4, 12, 10); #first arg is function, second arg is derivative

PyPlot.figure(1)
(x, y) = plt_data(b, .01, 2, 150)
(x1, y1) = plt_data(c, .01, 2, 150)

PyPlot.plot(x, y)
#PyPlot.plot(x1, y1)
#plt(b, .01, 1; label = "pdf")
#plt!(c, .01, 1; label = "cdf")
#plt(target[pind])
if false
for i = 151:160
    pind = i:i
    #prediction data
    Fx0 = reshape(data[pind, posc], 1, length(posc))
    x0 = reshape(data[pind, posx], 1, length(posx))
    b = y0 -> pdf(x0, Fx0, y0);
    c = y0 -> cdf(x0, Fx0, y0);
    figure(i)
    (x, y) = plt_data(b, .01, 1.5, 150)
    (x1, y1) = plt_data(c, .01, 1.5, 150)
    PyPlot.plot(x, y)
    PyPlot.plot(x1, y1)
    PyPlot.axvline(x=target[i])
end
end

if true

println("testing fast LOOCV cross validation")

m = 5
n = 8
plt, axs = PyPlot.subplots(5, 8)
#figure(1)
for j = 1:m*n
    (pdf, cdf, dpdf) = solve(btg1, validate = j);  
    #a = y0 -> dpdf(x0, Fx0, y0); 
    b = y0 -> pdf(x0, Fx0, y0);
    c = y0 -> cdf(x0, Fx0, y0);
    
    (x, y) = plt_data(b, .01, 1.2, 100)
    (x1, y1) = plt_data(c, .01, 1.2, 100)
    ind1 = Int64(ceil(j/8))
    ind2 = Int64(j - 8*(floor((j-.1)/8)))

    axs[ind1, ind2].plot(x, y)
    axs[ind1, ind2].plot(x1, y1)
    axs[ind1, ind2].axvline(x =  getLabel(btg1.trainingData)[j])
    #PyPlot.plot(x, y)
    #PyPlot.plot(x1, y1)
    #PyPlot.axvline(x =  getLabel(btg1.trainingData)[j])
end
for ax in axs
    ax.set(xlabel="x-label", ylabel="y-label")
end
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs
    ax.label_outer()
end
end

println("naive LOOCV")
m = 5
n = 8
plt, axs = PyPlot.subplots(5, 8)
#figure(1)
for j = 1:m*n
    x = getPosition(trainingData1)
    Fx = getCovariates(trainingData1)
    z = getLabel(trainingData1)
    

    cur_x = x[[1:j-1;j+1:end], :]
    cur_Fx = Fx[[1:j-1;j+1:end], :]
    cur_z = z[[1:j-1;j+1:end]]

    x_j = x[j:j, :]
    Fx_j = Fx[j:j, :]
    z_j = z[j:j, :]

    cur_td = trainingData(cur_x, cur_Fx, cur_z)

    btg1 = btg(cur_td, rangeθ, rangeλ) #quadtype = ["SparseGrid", "MonteCarlo"])

    (pdf, cdf, dpdf) = solve(btg1);  
    #a = y0 -> dpdf(x0, Fx0, y0); 
    b = y0 -> pdf(x_j, Fx_j, y0);
    c = y0 -> cdf(x_j, Fx_j, y0);
    
    (h1, h2) = plt_data(b, .01, 1.2, 100)
    (h3, h4) = plt_data(c, .01, 1.2, 100)
    ind1 = Int64(ceil(j/8))
    ind2 = Int64(j - 8*(floor((j-.1)/8)))

    axs[ind1, ind2].plot(h1, h2)
    axs[ind1, ind2].plot(h3, h4)
    axs[ind1, ind2].axvline(x =  z_j)
    #PyPlot.plot(x, y)
    #PyPlot.plot(x1, y1)
    #PyPlot.axvline(x =  getLabel(btg1.trainingData)[j])
end
for ax in axs
    ax.set(xlabel="x-label", ylabel="y-label")
end
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs
    ax.label_outer()
end
