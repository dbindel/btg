using Test
using Distributions
include("transforms.jl")
include("legpts.jl")
include("statistics.jl")

@testset "transforms.jl" begin
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
    @test int1D(f, -1, 1) ≈ 3.01706 atol = 1e-5
end

@testset "statistics" begin
    log = LogNormal(2, 1)
    y = x-> Distributions.cdf(log, x)-0.5
    med = median(log)
    @test bisection(y, 0, 15) ≈  med atol = 1e-3
end