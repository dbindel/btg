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
include("../computation/finitedifference.jl")                                      #             
include("../quadrature/quadrature.jl")                                             #
include("../transforms/transforms.jl")                                             #
include("../priors/priors.jl")                                                     #
include("../bayesopt/incremental.jl")                                              #
include("../kernels/kernel.jl")                                                    #
include("../datastructs.jl")                                                       #
include("../computation/buffers0.jl") #datastruct, kernel, incremental, quadrature #
include("../model0.jl") #buffers, datastructs, several auxiliary                   #
include("../computation/tdist.jl") #model0 and buffer0                             #
#################################################################################

df = DataFrame(CSV.File("../datasets/abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age
normalizing_constant = maximum(target)
target = target/normalizing_constant #normalization

#~~~~~~~~~~~~~~~~~~~~~~~~~~(to test or not to test)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_btg1 = false
test_btg2 = true
test_btg3 = true
test_btg4 = false #weird finite difference behavior when n >1000



##~~~~~~~~~~~~~~~~~~~~~~~~~~~~(btg1)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Attributes: 
# - single length scale
# - 2-dimensional covariates
# - 3-multidimensional location vectors

ind = 120:130
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
rangeλ = [0.5 5] #we will always used 1 range scale for lambda
btg1 = btg(trainingData1, rangeθ, rangeλ)
θ1 = btg1.nodesWeightsθ.nodes[1, 6] #pick some theta value which doubles as quadrature node
##################################################################################################

if (test_btg1)  #test comp_tdist 

    (dpdf, pdf, cdf) = comp_tdist(btg1, [θ1], [1.4]) 
    dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
    pdf_fixed = y0 -> pdf(x0, Fx0, y0)
    cdf_fixed = y0 -> cdf(x0, Fx0, y0)
    a = pdf_fixed
    b = cdf_fixed
    c = dpdf_fixed
    #rr = 8:16
    #h = abs.(b.(.5 .+ [2.718 ^(-i) for i=rr]) .- b.(.5) - [2.718 ^(-i) for i = rr] .* a(.5))
    #xx = [2.718 ^(-i) for i=rr]
    #lxx = log.(xx)
    #lh = log.(h)
    #polyfit(lxx, lh, 1)
    (_, _, _, pol1) = checkDerivative(a, c, 0.5, nothing, 5, 13, 10) #function first, then derivative
    (_, _, _, pol2) = checkDerivative(b, a, 0.5, nothing, 5, 13, 10) #function first, then derivative
    @testset "comp_tdist" begin
        @test coeffs(pol1)[end] ≈ 2 atol = 1e-1
        @test coeffs(pol2)[end] ≈ 2 atol = 1e-1
    end


if true #test bayesian predictive distribution (pdf, cdf, pdf_deriv)
    #rangeθ = [2.0 5; 4 7; 5 10]  #number of length scales is height of rangeθ

    (pdf, cdf, dpdf) = solve(btg1)  
    a = y0 -> dpdf(x0, Fx0, y0) 
    b = y0 -> pdf(x0, Fx0, y0)
    c = y0 -> cdf(x0, Fx0, y0)

    #println("Checking derivative of btg_pdf...")
    (_, _, plt1, pol1) = checkDerivative(b, a, 0.5, nothing, 8, 16, 10) #first arg is function, second arg is derivative
    (_, _, plt2, pol2) = checkDerivative(c, b, 0.5, nothing, 8, 16, 10) #first arg is function, second arg is derivative

    @testset "comp_btg_dist" begin
        @test coeffs(pol1)[end] ≈ 2 atol = 1e-1
        @test coeffs(pol2)[end] ≈ 2 atol = 1e-1
    end
    #display(plt1)
    #println(pol)
    #println("Plotting dpdf, pdf, and cdf...")
    if false
        plt(a, .1, 1, 100)
        plt(b, .1, 1, 100)
        plt!(c, .1, 1, 100, title = "PDF and CDF of Bayesian Predictive Distribution")
    end
end

end


if test_btg2 #test btg2
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~(btg2)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Attributes: 
# - 2 length scales
# - 3-dimensional covariate vectors
# - 2-dimensional location vectors
# polynomial mean basis

ind = 120:140
posx = [1;4] #
posc = 1:3
x = data[ind, posx] 
#choose a subset of variables to be regressors for the mean
POLYNOMIAL_BASIS_COVARIATES = true
if POLYNOMIAL_BASIS_COVARIATES
    Fx = hcat(ones(length(ind), 1), data[ind, 1:2])
else 
    Fx = data[ind, posc] 
end
 
y = float(target[ind])
pind = 10:10 #prediction index
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions

d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
if POLYNOMIAL_BASIS_COVARIATES
    Fx0 = hcat(ones(length(pind), 1), data[pind, 1:(length(posc)-1)])
else 
    Fx0 = reshape(data[pind, posc], 1, length(posc))
end
x0 = reshape(data[pind, posx], 1, length(posx)) 
rangeθ = [100.0 200; 200.0 400.0]
rangeλ = [0.5 5] #we will always used 1 range scale for lambda
btg2 = btg(trainingData1, rangeθ, rangeλ)
θ2 = btg2.nodesWeightsθ.nodes[1:2, 6] #pick some theta value which doubles as quadrature node
##################################################################################################

if true  #test comp_tdist for btg2
    (dpdf, pdf, cdf) = comp_tdist(btg2, θ2, [1.4]) 
    dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
    pdf_fixed = y0 -> pdf(x0, Fx0, y0)
    cdf_fixed = y0 -> cdf(x0, Fx0, y0)
    a = pdf_fixed
    b = cdf_fixed
    c = dpdf_fixed
    plot_multiple(a, b, c) #plot pdf, cdf, dpdf
    (_, _, plt1, pol1) = checkDerivative(a, c, 0.5, nothing, 3, 9, 10) #function first, then derivative
    (_, _, plt2, pol2) = checkDerivative(b, a, 0.5, nothing, 3, 9, 10) #function first, then derivative
    @testset "comp_tdist2" begin
        @test coeffs(pol1)[end] ≈ 2 atol = 3e-1
        @test coeffs(pol2)[end] ≈ 2 atol = 3e-1
    end
end

if true #test bayesian predictive distribution (pdf, cdf, pdf_deriv) for btg 2
    #rangeθ = [2.0 5; 4 7; 5 10]  #number of length scales is height of rangeθ
    (pdf, cdf, dpdf) = solve(btg2)  
    a = y0 -> dpdf(x0, Fx0, y0) 
    b = y0 -> pdf(x0, Fx0, y0)
    c = y0 -> cdf(x0, Fx0, y0)
    #println("Checking derivative of btg_pdf...")
    (_, _, plt1, pol1) = checkDerivative(b, a, 0.5, nothing, 8, 16, 10) #first arg is function, second arg is derivative
    (_, _, plt2, pol2) = checkDerivative(c, b, 0.5, nothing, 4, 10, 10) #first arg is function, second arg is derivative
    @testset "comp_btg_dist2" begin
        @test coeffs(pol1)[end] ≈ 2 atol = 3e-1
        @test coeffs(pol2)[end] ≈ 2 atol = 3e-1
    end
    #display(plt1)
    #println(pol)
    #println("Plotting dpdf, pdf, and cdf...")
    if false
        plt(a, .1, 1, 100)
        plt(b, .1, 1, 100)
        plt!(c, .1, 1, 100, title = "PDF and CDF of Bayesian Predictive Distribution")
    end
end


if true #test derivatives of location, and derivatives of auxiliary functions w.r.t location
    ### test jacobian and Hessian of C, D, m, etc.
    @testset "location derivs in btg2" begin  
        Fx0 = x0 -> hcat([1], reshape(x0, 1, length(x0))) #linear polynomial basis
        y0 = .5
        (_, _, _, cdf_prime_loc) = comp_tdist(btg2, θ2, [1.4])      
        A = x0 -> cdf_prime_loc(reshape(x0, 1, length(x0)), Fx0(x0), y0)[1]
        B = x0 -> cdf_prime_loc(reshape(x0, 1, length(x0)), Fx0(x0), y0)[2]
        C = x0 -> cdf_prime_loc(reshape(x0, 1, length(x0)), Fx0(x0), y0)[3]
        init_x = x[1:1, :] + rand(1, length(posx)) .* 1e-3
        #println(size(init_x))
        (_, _, plt1, pol1) = checkDerivative(A, B, init_x, nothing, 2, 5, 10) #first arg is function, second arg is derivative
        @test coeffs(pol1)[end] > 2 - 3e-1
        println("loc deriv pol1: ", pol1)
        display(plt1)
        (_, _, plt2, pol2) = checkDerivative(A, B, init_x, C, 2, 6, 10) #first arg is function, second arg is derivative
        @test coeffs(pol2)[end] > 3 - 3e-1
        println("loc deriv pol2: ", pol2)
        display(plt2)
    end
end


end
if test_btg3 #test btg3
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~(btg3)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Attributes: 
# - 3 length scales
# - 3-D covariate vectors
# - 2-D location vectors

ind = 120:135
posx = [1;3;4] #
posc = 1:3
x = data[ind, posx] 
#choose a subset of variables to be regressors for the mean
Fx = data[ind, posc] 
y = float(target[ind])
pind = 10:10 #prediction index
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions


d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
Fx0 = reshape(data[pind, posc], 1, length(posc))
x0 = reshape(data[pind, posx], 1, length(posx)) 
rangeθ = [100.0 200; 200.0 400.0; 100.0 1000.0]
rangeλ = [0.5 5] #we will always used 1 range scale for lambda
btg3 = btg(trainingData1, rangeθ, rangeλ)
θ3 = btg3.nodesWeightsθ.nodes[1:3, 6] #pick some theta value which doubles as quadrature node
##################################################################################################

if true  #test comp_tdist for btg3
    (dpdf, pdf, cdf) = comp_tdist(btg3, θ3, [1.4]) 
    dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
    pdf_fixed = y0 -> pdf(x0, Fx0, y0)
    cdf_fixed = y0 -> cdf(x0, Fx0, y0)
    a = pdf_fixed
    b = cdf_fixed
    c = dpdf_fixed
    (_, _, _, pol1) = checkDerivative(a, c, 0.5, nothing, 4, 12, 10) #function first, then derivative
    (_, _, _, pol2) = checkDerivative(b, a, 0.5, nothing, 4, 12, 10) #function first, then derivative
    @testset "comp_tdist3" begin
        @test coeffs(pol1)[end] ≈ 2 atol = 3e-1
        @test coeffs(pol2)[end] ≈ 2 atol = 3e-1
    end
end

if true #test bayesian predictive distribution (pdf, cdf, pdf_deriv) for btg 3
    #rangeθ = [2.0 5; 4 7; 5 10]  #number of length scales is height of rangeθ
    (pdf, cdf, dpdf) = solve(btg3)  
    a = y0 -> dpdf(x0, Fx0, y0) 
    b = y0 -> pdf(x0, Fx0, y0)
    c = y0 -> cdf(x0, Fx0, y0)
    #println("Checking derivative of btg_pdf...")
    (_, _, plt1, pol1) = checkDerivative(b, a, 0.5, nothing, 4, 12, 10) #first arg is function, second arg is derivative
    (_, _, plt2, pol2) = checkDerivative(c, b, 0.5, nothing, 4, 12, 10) #first arg is function, second arg is derivative
    @testset "comp_btg_dist3" begin
        @test coeffs(pol1)[end] ≈ 2 atol = 1e-1
        @test coeffs(pol2)[end] ≈ 2 atol = 1e-1
    end
    #display(plt1)
    #println(pol)
    #println("Plotting dpdf, pdf, and cdf...")
    if false
        plt(a, .1, 1, 100)
        plt(b, .1, 1, 100)
        plt!(c, .1, 1, 100, title = "PDF and CDF of Bayesian Predictive Distribution")
    end
end
end


if test_btg4 #test btg4
    println("Runnings tests for BTG4....")
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~(btg4)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Attributes: 
    # - 3 length scales
    # - 3-D covariate vectors
    # - 2-D location vectors
    #- use Monte-Carlo integration
    
    ind = 120:1120
    posx = [1;3;4] #
    posc = 1:3
    x = data[ind, posx] 
    #choose a subset of variables to be regressors for the mean
    Fx = data[ind, posc] 
    y = float(target[ind])
    pind = 10:10 #prediction index
    trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
    
    d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
    Fx0 = reshape(data[pind, posc], 1, length(posc))
    x0 = reshape(data[pind, posx], 1, length(posx)) 
    rangeθ = [100.0 200; 200.0 400.0; 100.0 1000.0]
    rangeλ = [0.5 5] #we will always used 1 range scale for lambda
    btg4 = btg(trainingData1, rangeθ, rangeλ; quadtype = ["MonteCarlo", "Gaussian"])
    θ4 = btg4.nodesWeightsθ.nodes[1:3, 6] #pick some theta value which doubles as quadrature node
    ##################################################################################################
    
    if true  #test comp_tdist for btg4
        (dpdf, pdf, cdf) = comp_tdist(btg4, θ4, [1.4]) 
        dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
        pdf_fixed = y0 -> pdf(x0, Fx0, y0)
        cdf_fixed = y0 -> cdf(x0, Fx0, y0)
        a = pdf_fixed
        b = cdf_fixed
        c = dpdf_fixed
        (_, _, plt1, pol1) = checkDerivative(a, c, 0.5, nothing, -20, -10, 20) #function first, then derivative
        (_, _, plt2, pol2) = checkDerivative(b, a, 0.5, nothing, -15, -10, 20) #function first, then derivative
        @testset "comp_tdist4" begin
            @test coeffs(pol1)[end] ≈ 2 atol = 3e-1
            @test coeffs(pol2)[end] ≈ 2 atol = 3e-1
        end
    end
    
    if true #test bayesian predictive distribution (pdf, cdf, pdf_deriv) for btg 4
        #rangeθ = [2.0 5; 4 7; 5 10]  #number of length scales is height of rangeθ
        (pdf, cdf, dpdf) = solve(btg4)  
        a = y0 -> dpdf(x0, Fx0, y0) 
        b = y0 -> pdf(x0, Fx0, y0)
        c = y0 -> cdf(x0, Fx0, y0)
        #println("Checking derivative of btg_pdf...")
        (_, _, plt1, pol1) = checkDerivative(b, a, 0.5, nothing, 4, 12, 10) #first arg is function, second arg is derivative
        (_, _, plt2, pol2) = checkDerivative(c, b, 0.5, nothing, 4, 12, 10) #first arg is function, second arg is derivative
        @testset "comp_btg_dist4" begin
            @test coeffs(pol1)[end] ≈ 2 atol = 3e-1
            @test coeffs(pol2)[end] ≈ 2 atol = 3e-1
        end
        #display(plt1)
        #println(pol)
        #println("Plotting dpdf, pdf, and cdf...")
        if false
            plt(a, .1, 1, 100)
            plt(b, .1, 1, 100)
            plt!(c, .1, 1, 100, title = "PDF and CDF of Bayesian Predictive Distribution")
        end
    end
    end