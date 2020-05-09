#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(The BTG Program)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################################################################
#copy this box into a file to use the BTG program                              
using LinearAlgebra                                                           
using Test                                                                                    
using Distributions                                                           
using DataFrames                                                                    
using CSV                                                                         
using Polynomials   
include("../covariatefun.jl")
include("../range_selector.jl")
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
using Plots
using TimerOutputs
const to = TimerOutput()
###
### This file is used to call pdf, cdf, dpdf, cdf_derivs 100s of times to see how well the code scales. These operations
### are the workhorses of other BTG routines.
###
###

df = DataFrame(CSV.File("../datasets/abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age
normalizing_constant = maximum(target)
target = target/normalizing_constant #normalization

ind = 120:520
posx = 1:7 #
posc = 1:7
x = data[ind, posx] 
@info "recommended theta range: ", select_single_theta_range(x)
#choose a subset of variables to be regressors for the mean
POLYNOMIAL_BASIS_COVARIATES = true
if POLYNOMIAL_BASIS_COVARIATES
    Fx = hcat(ones(length(ind), 1), x)
else 
    Fx = data[ind, posc] 
end
 
y = float(target[ind])
pind = 10:10 #prediction index
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions

d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)   
x0 = reshape(data[pind, posx], 1, length(posx)) 
if POLYNOMIAL_BASIS_COVARIATES
    Fx0 = hcat(1, x0)
else 
    Fx0 = reshape(data[pind, posc], 1, length(posc))
end
rangeθ = [10 1000.0]
rangeλ = [0.2 2] #we will always used 1 range scale for lambda
println("profiling btg initialization")
@timeit to "init btg2" btg2 = btg(trainingData1, rangeθ, rangeλ);
θ2 = btg2.nodesWeightsθ.nodes[6] #pick some theta value which doubles as quadrature node
λ2 = btg2.nodesWeightsλ.nodes[1]
@timeit to "eval btg2" begin 
    if false
        println("profiling btg evaluations")
        y0 =1.1
        x0 = [0.8 0.02 0.4 0.3 0.5 0.2 0.4]
        @timeit to "get pdf, cdf, dpdf" (pdf, cdf, dpdf, cdf_grad_us, cdf_hess_us, quantInfo) = solve(btg2; derivatives = true)
        A = au -> pdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
        B = au -> cdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
        C = au -> dpdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
        D = au -> cdf_grad_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
        E = au -> cdf_hess_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
        init = hcat(y0, x0)
        for i = 1:100
            println("btg evals iteration $i")
            cur = init + rand(1, length(init)) .* .05 
            #@timeit to "pdf eval" C(cur)
            @timeit to "cdf eval" D(cur)
        end
    end
    if true
        println("profiling btg derivatives")
        y0 =1.1
        x0 = [0.8 0.02 0.4 0.3 0.5 0.2 0.4]
        @timeit to "get pdf, cdf, dpdf" (pdf, cdf, dpdf, cdf_grad_us, cdf_hess_us, quantInfo) = solve(btg2; derivatives = true)
        A = au -> cdf_grad_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
        B = au -> cdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
        C = au -> cdf_hess_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
        init = hcat(y0, x0)
        @timeit to "checkDerivative" (_, _, plt1, pol1) = checkDerivative(B, A, init, nothing, 5, 11, 10) #first arg is function, second arg is derivative
        @test coeffs(pol1)[end] > 2 - 3e-1
        println("pol1:",pol1)
        (_, _, plt2, pol2) = checkDerivative(B, A, init, C, 5, 11, 10) #first arg is function, second arg is derivative
        println("pol1:",pol2)
        @test coeffs(pol2)[end]> 3 - 3e-1
    end
end