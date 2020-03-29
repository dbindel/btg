using Test
using Distributions
include("../kernels/kernel.jl")
include("../computation/finitedifference.jl")

@testset "kernel.jl" begin

a = (tau, x) -> exp.(-x*tau/2)
da = (tau, x) -> -tau/2 * exp.(-x*tau/2)
gg = x -> a(2, x)
dgg =  x -> da(2, x)
(h, A, plt1, poly, fis1) = checkDerivative(gg, dgg, 1.0, nothing, 1, 2, 10)
#println(poly)
#display(plt1)

#dsdf = RBF()
dsdf = Gaussian()
ff = y -> dsdf(2, y)
dff =  y -> partial_θ(dsdf, 2, y)
(h, A, plt1, poly, fis2, vals, debugvals) = checkDerivative(ff, dff, 1.0, nothing, 1, 2, 10)
println(poly)
#display(plt1)

Plots.plot(vals[:, 1], seriestype = :scatter)
for i =2:10
    println("here")
    Plots.plot!(vals[:, i], seriestype = :scatter)
end

#Plots.plot(fis1')
#Plots.plot(fis2')

#plt(ff, .01, 10)
#plt!(dff, .01, 10)
#plt!(gg, .01, 10)
#plt!(dgg, .01, 10)

end

@testset "kernel.jl" begin
    @test invBoxCox(boxCox(2, 3), 3) ≈ 2.0 atol = 1e-15
    @test invBoxCox(boxCox(2, 1), 1) ≈ 2.0 atol = 1e-15
    @test boxCox(invBoxCox(2, 3), 3) ≈ 2.0 atol = 1e-15
    @test boxCox(invBoxCox(2, 1), 1) ≈ 2.0 atol = 1e-15
    @test boxCox(invBoxCox([1 2 3], 1), 1) ≈ [1 2 3] atol = 1e-15
    @test boxCox(invBoxCox([1; 2], 1), 1) ≈ [1;2] atol = 1e-15
    @test boxCox(invBoxCox(2., -0.45), -0.45) ≈ 2.0 atol = 1e-15
    @test invBoxCox(boxCox(2., -2), -2) ≈ 2.0 atol = 1e-15
end

@testset "legpts.jl" begin
    f = x -> exp(x)+x^2    
    @test_broken int1D(f, -1, 1) ≈ 3.01706 atol = 1e-5
end

@testset "statistics" begin
    log = LogNormal(2, 1)
    y = x-> Distributions.cdf(log, x)-0.5
    med = median(log)
    @test_broken bisection(y, 0, 15) ≈  med atol = 1e-3
end

@testset ""