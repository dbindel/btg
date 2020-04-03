using Test
using Distributions

include("../kernels/kernel.jl")
include("../computation/finitedifference.jl")
include("../computation/tdist.jl")
include("../transforms/transforms.jl")
include("../datastructs.jl")

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