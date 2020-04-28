#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~(The BTG Program)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#################################################################################
#copy this box into a file to use the BTG program                              
using LinearAlgebra                                                           
using Test                                                                                    
using Distributions                                                           
using DataFrames                                                                    
using CSV                                                                         
using Polynomials   
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

df = DataFrame(CSV.File("../datasets/abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age
normalizing_constant = maximum(target)
target = target/normalizing_constant #normalization

#~~~~~~~~~~~~~~~~~~~~~~~~~~(to test or not to test)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_btg1 = false
test_btg2 = true #used to test derivatives of cdf w.r.t augmented label-location vector [u, s]
test_btg3 = false
test_btg4 = false #weird finite difference behavior when n >1000

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~(btg1)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Attributes: 
# - single length scale
# - 2-dimensional covariates
# - 3-multidimensional location vectors

ind = 120:125
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
λ1 = btg1.nodesWeightsλ.nodes[3]
##################################################################################################

if (test_btg1)  #test comp_tdist 
    (dpdf, pdf, cdf) = comp_tdist(btg1, θ1, λ1) 
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
    
    
    #buf = btg1.test_buffer_dict[θ1]
    #println(buf.Eθ)
    #unpack(buf)
    
    
    #a = unpack(btg1.test_buffer_dict[θ1])
    #println("unpacked val in test_tdist: ", a)
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
    (_, _, plt1, pol1) = checkDerivative(b, a, 0.5, nothing, 4, 12, 10) #first arg is function, second arg is derivative
    (_, _, plt2, pol2) = checkDerivative(c, b, 0.5, nothing, 4, 12, 10) #first arg is function, second arg is derivative

    @testset "comp_btg_dist" begin
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


if test_btg2 #test btg2
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~(btg2)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Attributes: 
# - 2 length scales
# - 3-dimensional covariate vectors
# - 2-dimensional location vectors
# polynomial mean basis

ind = 120:140
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
btg2 = btg(trainingData1, rangeθ, rangeλ);
θ2 = btg2.nodesWeightsθ.nodes[6] #pick some theta value which doubles as quadrature node
λ2 = btg2.nodesWeightsλ.nodes[1]
##################################################################################################

if false  #test comp_tdist for btg2
    (dpdf, pdf, cdf) = comp_tdist(btg2, θ2, λ2) 
    dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
    pdf_fixed = y0 -> pdf(x0, Fx0, y0)
    cdf_fixed = y0 -> cdf(x0, Fx0, y0)
    a = pdf_fixed
    b = cdf_fixed
    c = dpdf_fixed
    #plot_multiple(a, b, c) #plot pdf, cdf, dpdf
    (_, _, plt1, pol1) = checkDerivative(a, c, 0.5, nothing, 3, 9, 10) #function first, then derivative
    (_, _, plt2, pol2) = checkDerivative(b, a, 0.5, nothing, 3, 9, 10) #function first, then derivative
    @testset "comp_tdist2" begin
        @test coeffs(pol1)[end] ≈ 2 atol = 3e-1
        @test coeffs(pol2)[end] ≈ 2 atol = 3e-1
    end
end

if false #test bayesian predictive distribution (pdf, cdf, pdf_deriv) for btg 2
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

if true
    #if true #test derivatives of new function 
        println("test derivative of cdf/cdf-related quantities w.r.t augmented vector [u, s]...\n")
        #v is augmented vector 
            function linear_polynomial_basis(x0)#linear polynomial basis
                return hcat([1], reshape(x0, 1, length(x0)))
            end
            #val = [0.45, 0.375, 0.115, 0.4105]
            (dpdf, pdf, cdf, cdf_grad_us, cdf_hess_us) = comp_tdist(btg2, θ2, λ2)    
           # A = v -> cdf((tmp = v[2:end]; reshape(tmp, 1, length(tmp))), (tmp = Fx0(v[2:end]); reshape(tmp, 1, length(tmp))), v[1])
            #B = v -> cdf_grad_us(v[2:end], Fx0(v[2:end]), v[1])
            #y=.26
            if false
            @testset "augmented comp_tdist test1" begin #close to actual data point
                y0 = .26
                x0 = [0.35 0.275 0.1 0.2355 0.0895 0.0585 0.08]
                A = au -> cdf_grad_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                B = au -> cdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                C = au -> cdf_hess_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                init = hcat(y0, x0)
                (_, _, plt1, pol1) = checkDerivative(B, A, init, nothing, 5, 11, 10) #first arg is function, second arg is derivative
                @test coeffs(pol1)[end] > 2 - 3e-1
                println("pol1:",pol1)
                (_, _, plt2, pol2) = checkDerivative(B, A, init, C, 5, 11, 10) #first arg is function, second arg is derivative
                println("pol1:",pol2)
                @test coeffs(pol2)[end]> 3 - 3e-1
            end
                
            @testset "augmented comp_tdist test2 " begin #perturbation to y
                y0 = .8
                x0 = [0.35 0.275 0.1 0.2355 0.0895 0.0585 0.08]
                A = au -> cdf_grad_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                B = au -> cdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                C = au -> cdf_hess_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                init = hcat(y0, x0)
                (_, _, plt1, pol1) = checkDerivative(B, A, init, nothing, 5, 11, 10) #first arg is function, second arg is derivative
                @test coeffs(pol1)[end] > 2 - 3e-1
                println("pol1:",pol1)
                (_, _, plt2, pol2) = checkDerivative(B, A, init, C, 5, 11, 10) #first arg is function, second arg is derivative
                println("pol1:",pol2)
                @test coeffs(pol2)[end]> 3 - 3e-1
            end

            @testset "augmented comp_tdist test3 " begin #perturbation to y and x
                y0 = .8
                x0 = [0.3 0.4 0.05 0.3 0.03 0.05 0.15]
                A = au -> cdf_grad_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                B = au -> cdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                C = au -> cdf_hess_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                init = hcat(y0, x0)
                (_, _, plt1, pol1) = checkDerivative(B, A, init, nothing, 5, 11, 10) #first arg is function, second arg is derivative
                @test coeffs(pol1)[end] > 2 - 3e-1
                println("pol1:",pol1)
                (_, _, plt2, pol2) = checkDerivative(B, A, init, C, 5, 11, 10) #first arg is function, second arg is derivative
                println("pol1:",pol2)
                @test coeffs(pol2)[end]> 3 - 3e-1
            end

            @testset "augmented comp_tdist test4" begin #major perturbation (.8 in all x coords) to y and x; y outside [0, 1]
                y0 = 1.2
                x0 = [0.8 0.8 0.8 0.8 0.8 0.8 0.8]
                A = au -> cdf_grad_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                B = au -> cdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                C = au -> cdf_hess_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                init = hcat(y0, x0)
                (_, _, plt1, pol1) = checkDerivative(B, A, init, nothing, 5, 11, 10) #first arg is function, second arg is derivative
                @test coeffs(pol1)[end] > 2 - 3e-1
                println("pol1:",pol1)
                (_, _, plt2, pol2) = checkDerivative(B, A, init, C, 5, 11, 10) #first arg is function, second arg is derivative
                println("pol1:",pol2)
                @test coeffs(pol2)[end]> 3 - 3e-1
            end
        end
        if true
            println("test derivatives of average augmented cdf_deriv...\n ")
            @testset "btg augmented cdf_function test 1" begin
                y0 = 0.4
                x0 = [0.3 0.4 0.05 0.3 0.03 0.05 0.15]
                (pdf, cdf, dpdf, cdf_grad_us, cdf_hess_us, quantInfo) = solve(btg2; derivatives = true)
                A = au -> cdf_grad_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                B = au -> cdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                C = au -> cdf_hess_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                init = hcat(y0, x0)
                (_, _, plt1, pol1) = checkDerivative(B, A, init, nothing, 5, 11, 10) #first arg is function, second arg is derivative
                @test coeffs(pol1)[end] > 2 - 3e-1
                println("pol1:",pol1)
                (_, _, plt2, pol2) = checkDerivative(B, A, init, C, 5, 11, 10) #first arg is function, second arg is derivative
                println("pol1:",pol2)
                @test coeffs(pol2)[end]> 3 - 3e-1
            end
            @testset "btg augmented cdf_function test 2" begin
                y0 =1.1
                x0 = [0.8 0.02 0.4 0.3 0.5 0.2 0.4]
                (pdf, cdf, dpdf, cdf_grad_us, cdf_hess_us, quantInfo) = solve(btg2; derivatives = true)
                A = au -> cdf_grad_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                B = au -> cdf(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                C = au -> cdf_hess_us(au[2:end], linear_polynomial_basis(au[2:end]), au[1])
                init = hcat(y0, x0)
                (_, _, plt1, pol1) = checkDerivative(B, A, init, nothing, 5, 11, 10) #first arg is function, second arg is derivative
                @test coeffs(pol1)[end] > 2 - 3e-1
                println("pol1:",pol1)
                (_, _, plt2, pol2) = checkDerivative(B, A, init, C, 5, 11, 10) #first arg is function, second arg is derivative
                println("pol1:",pol2)
                @test coeffs(pol2)[end]> 3 - 3e-1
            end
        end
        #end
    #end
    end
if false
#if true #test derivatives of new function 
    println("test derivative of cdf/cdf-related quantities computed using comp_tdist w.r.t s")
    #v is augmented vector 
        Fx0 = x0 -> hcat([1], reshape(x0, 1, length(x0))) #linear polynomial basis
        #val = [0.45, 0.375, 0.115, 0.4105]
        (dpdf, pdf, cdf, cdf_grad_us, cdf_hess_us) = comp_tdist(btg2, θ2, λ2)    
       # A = v -> cdf((tmp = v[2:end]; reshape(tmp, 1, length(tmp))), (tmp = Fx0(v[2:end]); reshape(tmp, 1, length(tmp))), v[1])
        #B = v -> cdf_grad_us(v[2:end], Fx0(v[2:end]), v[1])
        y=.26
        A = x0 -> cdf_grad_us(x0, Fx0(x0), y)
        B = x0 -> cdf(x0, Fx0(x0), y)
        C = x0 -> cdf_hess_us(x0, Fx0(x0), y)
        #init_v = (tmp = hcat(0.5, x[1:1, :] + rand(1, length(posx)) .* 1e-1); reshape(tmp, 1, length(tmp)))
        #init_v = [0.32, 0.45, 0.375, 0.115, 0.4105]
        init_v =  [0.35 0.275 0.1 0.2355 0.0895 0.0585 0.08]
        if false
            #cdf_fixed = y -> cdf([0.45, 0.375, 0.115, 0.4105], Fx0([0.45, 0.375, 0.115, 0.4105]), y)
            #cdf_fixed 
            #plt(cdf_fixed, .01, 1, 100)
        end 
        (_, _, plt1, pol1) = checkDerivative(B, A, init_v, nothing, 4, 10, 10) #first arg is function, second arg is derivative
        #@test coeffs(pol1)[end] > 2 - 3e-1
        println(pol1)
        (_, _, plt2, pol2) = checkDerivative(B, A, init_v, C, 3, 13, 10); #first arg is function, second arg is derivative
        #@test coeffs(pol1)[end] > 2 - 3e-1
        println(pol2)

        if false #temporary test for second derivative of cdf w.r.t x
        A = y -> cdf_hess_us(init_v, Fx0(init_v), y)[1]
        B = y -> cdf_hess_us(init_v, Fx0(init_v), y)[2]
        C = y -> cdf_hess_us(init_v, Fx0(init_v), y)[3]
        (_, _, plt1, pol1) = checkDerivative(A, B, 0.26, nothing, 4, 10, 10) #first arg is function, second arg is derivative
        #@test coeffs(pol1)[end] > 2 - 3e-1
        println(pol1)
        (_, _, plt2, pol2) = checkDerivative(B, C, 0.26, nothing, 3, 13, 10); #first arg is function, second arg is derivative
        #@test coeffs(pol1)[end] > 2 - 3e-1
        println(pol2)
        end
    #end
#end
end
if false
#if true #test derivatives of new function 
println("test derivative of cdf w.r.t value/label u")
#v is augmented vector 
    Fx0 = x0 -> hcat([1], reshape(x0, 1, length(x0))) #linear polynomial basis
    val = [.425 .3 .1 .35 .15 .09 .15]
    (pdf, dpdf, cdf, cdf_grad_us) = comp_tdist(btg2, θ2, λ2)    
   # A = v -> cdf((tmp = v[2:end]; reshape(tmp, 1, length(tmp))), (tmp = Fx0(v[2:end]); reshape(tmp, 1, length(tmp))), v[1])
    #B = v -> cdf_grad_us(v[2:end], Fx0(v[2:end]), v[1])
    A = y -> cdf_grad_us(val, Fx0(val), y)
    B = y -> cdf(val, Fx0(val), y)
    
    #init_v = (tmp = hcat(0.5, x[1:1, :] + rand(1, length(posx)) .* 1e-1); reshape(tmp, 1, length(tmp)))
    #init_v = [0.32, 0.45, 0.375, 0.115, 0.4105]
    init_v = 0.3
    if false
        #cdf_fixed = y -> cdf([0.45, 0.375, 0.115, 0.4105], Fx0([0.45, 0.375, 0.115, 0.4105]), y)
        plt(B, .01, 1, 100)
    end 
    (_, _, plt1, pol1) = checkDerivative(B, A, init_v, nothing, 4, 15, 10) #first arg is function, second arg is derivative
    #@test coeffs(pol1)[end] > 2 - 3e-1
    println(pol1)
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
λ3 = btg3.nodesWeightsλ.nodes[3]
##################################################################################################

if true  #test comp_tdist for btg3
    (dpdf, pdf, cdf) = comp_tdist(btg3, θ3, λ3) 
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
    
    ind = 120:250
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
    λ4 = btg4.nodesWeightsλ.nodes[3]
    ##################################################################################################
    
    if true  #test comp_tdist for btg4
        (dpdf, pdf, cdf) = comp_tdist(btg4, θ4, λ4) 
        dpdf_fixed = y0 -> dpdf(x0, Fx0, y0) 
        pdf_fixed = y0 -> pdf(x0, Fx0, y0)
        cdf_fixed = y0 -> cdf(x0, Fx0, y0)
        a = pdf_fixed
        b = cdf_fixed
        c = dpdf_fixed
        (_, _, plt1, pol1) = checkDerivative(a, c, 0.5, nothing, 4, 12, 20) #function first, then derivative
        (_, _, plt2, pol2) = checkDerivative(b, a, 0.5, nothing, 4, 12, 20) #function first, then derivative
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