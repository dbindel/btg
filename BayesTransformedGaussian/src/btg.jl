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
include("validation/det_loocv.jl")                                             
include("validation/loocv.jl")                                             
include("quadrature/quadrature.jl")                                             
include("transforms/transforms.jl")                                             
include("priors/priors.jl")                                                     
include("bayesopt/incremental.jl")                                              
include("kernels/kernel.jl")                                                    
include("datastructs.jl")                                                       
include("computation/buffers0.jl") #datastruct, kernel, incremental, quadrature 
include("model0.jl") #buffers, datastructs, several auxiliary   
include("computation/finitedifference.jl")                                                                
include("computation/tdist.jl") #model0 and buffer0                             
#################################################################################