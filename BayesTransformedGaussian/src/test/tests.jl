using Test
using Distributions
using DataFrames
using CSV

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

const ind = 1:5
const x = data[ind, :] 
#choose a subset of variables to be regressors for the mean
const Fx = data[ind, 1:1] 
const y = float(target[ind])

trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions

@testset "kernel.jl" begin
    @test BoxCox()(3, 1) ≈ 2.0 atol = 1e-9
    @test BoxCox()(inverse(BoxCox(), 2 ,3), 3) ≈ 2 atol = 1e-9
    #@test invBoxCox(boxCox(2, 3), 3) ≈ 2.0 atol = 1e-15
    #@test invBoxCox(boxCox(2, 1), 1) ≈ 2.0 atol = 1e-15
    #@test boxCox(invBoxCox(2, 3), 3) ≈ 2.0 atol = 1e-15
    #@test boxCox(invBoxCox(2, 1), 1) ≈ 2.0 atol = 1e-15
    #@test boxCox(invBoxCox([1 2 3], 1), 1) ≈ [1 2 3] atol = 1e-15
    #@test boxCox(invBoxCox([1; 2], 1), 1) ≈ [1;2] atol = 1e-15
    #@test boxCox(invBoxCox(2., -0.45), -0.45) ≈ 2.0 atol = 1e-15
    #@test invBoxCox(boxCox(2., -2), -2) ≈ 2.0 atol = 1e-15
end

@testset "datastructs.jl" begin
    x = newExtensibleTrainingData(5, 2, 3)
    @test x.d == 2
    @test x.p == 3
    @test x.n == 0
    o = [1.0 2.0; 3.0 4.0]
    p = [1.0 2.0 3.0; 4.0 5.0 6.0]
    z = [1.0, 2.0]
    update!(x, o, p, z)
    @test x.x[1:2, :] == o
    @test x.Fx[1:2, :] == p
    @test x.y[1:2] == z
    @test x.n == 2
    o2 = [1.0 2.0; 3.0 4.0; 4.0 5.0]
    p2 = [1.0 2.0 3.0; 4.0 5.0 6.0; 1.0 2.0 3.0]
    z2 = [1.0;2.0; 3.0]
    update!(x, o2, p2, z2) 
    @test x.x[1:5, :] == vcat(o, o2)
    @test x.Fx[1:5, :] == vcat(p, p2)
    @test x.y[1:5] == vcat(z, z2)
    @test x.n == 5
    o3 = [1.0 2.0]
    p3 = [1.0 2.0 3.0]
    z3 = [3.0]
   # @test_broken BoundsError update!(x, o3, p3, z3) 
end

@testset "tdist.jl" begin
    d = getDimension(trainingData); n = getNumPts(trainingData); p = getCovDimension(trainingData)
    rangeθ = [2.0 5; 4 7; 10 20; 1 3; 1 4]  
    rangeλ = [0.5 5]
    btg1 = btg(trainingData1, rangeθ, rangeλ)
end


#@testset "kernel.jl" begin
#a = (tau, x) -> exp.(-x*tau/2)
#da = (tau, x) -> -tau/2 * exp.(-x*tau/2)
#gg = x -> a(2, x)
#dgg =  x -> da(2, x)
#(h, A, plt1, poly, fis1) = checkDerivative(gg, dgg, 1.0, nothing, 1, 2, 10)
#println(poly)
#display(plt1)

#dsdf = RBF()
#dsdf = Gaussian()
#ff = y -> dsdf(2, y)
#dff =  y -> partial_θ(dsdf, 2, y)
#(h, A, plt1, poly, fis2, vals, debugvals) = checkDerivative(ff, dff, 1.0, nothing, 1, 2, 10)
#println(poly)
#display(plt1)
#Plots.plot(vals[:, 1], seriestype = :scatter)
#for i =2:10
#    println("here")
#    Plots.plot!(vals[:, i], seriestype = :scatter)
#end
#Plots.plot(fis1')
#Plots.plot(fis2')
#plt(ff, .01, 10)
#plt!(dff, .01, 10)
#plt!(gg, .01, 10)
#plt!(dgg, .01, 10)
#end