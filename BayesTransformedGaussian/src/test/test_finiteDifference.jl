using Test
include("../computation/finitedifference.jl")


@testset "R^1 -> R^1" begin
    f = x -> x^2*exp(x)
    jacf = x -> 2x*exp(x) + x^2*exp(x)
    (_, _, _, pol) = checkDerivative(f, jacf, .64, nothing) #Float64 -> Float64
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1
    f = x -> x[1]^2*exp(x[1])
    jacf = x -> 2x[1]*exp(x[1])+x[1]^2*exp(x[1])
    (_, _, _, pol) = checkDerivative(f, jacf, [.64], nothing) #Array{Float64, 1} -> Float64
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1
    f = x -> [x[1]^2*exp(x[1])]
    jacf = x -> [2x[1]*exp(x[1])+x[1]^2*exp(x[1])]
    (_, _, _, pol) = checkDerivative(f, jacf, [.64], nothing) #Array{Float64, 1} -> Array{Float64, 1}
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1
    
   
    #f = x -> [x^2*exp(x)]
    #jacf = x -> [2x*exp(x) + x^2*exp(x)]
    #(_, _, _, pol) = checkDerivative(f, jacf, .64, nothing) #Float64 -> Array{Float64, 1}
    f = x -> x^2*exp(x)
    jacf = x -> 2x*exp(x) + x^2*exp(x)
    hess_f = x -> 4x*exp(x) + 2*exp(x) + x^2*exp(x)
    (_, _, _, pol) = checkDerivative(f, jacf, .64, hess_f) #Float64 -> Float64
    @test coeffs(pol)[end] ≈ 3 atol = 3e-1
end


@testset "R^3 -> R^1" begin
    ff = xx ->  xx[1]^2 * sin(xx[2]*xx[3])
    jacf = xx -> [2*xx[1]*sin(xx[2]*xx[3])  xx[1]^2*xx[3]*cos(xx[2]*xx[3])  xx[1]^2*xx[2]*cos(xx[2]*xx[3])]
    (_, _, _, pol) = checkDerivative(ff, jacf, [.4 .5 .7], nothing) #initial guess has shape (3, 1)

    @test coeffs(pol)[end] ≈ 2 atol = 3e-1
    (_, _, _, pol) = checkDerivative(ff, jacf, [.4, .5, .7], nothing) #initial guess has shape (3,)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1
    (_, _, _, pol) = checkDerivative(ff, jacf, reshape([.4, .5, .7], 1, 3), nothing) #initial guess has shape (1, 3)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1

    #check Hessian   
    hess_f = x -> [2sin(x[2]x[3]) 2x[1]x[3]cos(x[2]x[3]) 2x[1]x[2]cos(x[2]x[3]) ;
                    2x[1]x[3]cos(x[2]x[3]) -x[1]^2 * x[3]^2 * sin(x[2]x[3]) x[1]^2*cos(x[2]x[3]) - x[1]^2*x[2]x[3]sin(x[2]x[3]);
                    2x[1]x[2]cos(x[2]x[3]) x[1]^2*cos(x[2]x[3]) - x[1]^2*x[2]x[3]sin(x[2]x[3]) -x[1]^2*x[2]^2 * sin(x[2]x[3])]
    (_, _, plt1, pol) = checkDerivative(ff, jacf, reshape([.4, .5, .7], 1, 3), hess_f, 2, 6) #initial guess has shape (1, 3)
    #display(plt1)
    @test coeffs(pol)[end] ≈ 3 atol = 3e-1
end

@testset "R^1 -> R^3" begin
    f = x -> [x;sin(x);log(x)]
    jacf = x -> [1;cos(x);1/x]
    (_, _, _, pol) = checkDerivative(f, jacf, .4, nothing)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1 #input is scalar

    g = x -> [x[1];sin(x[1]);log(x[1])]
    jacg = x -> [1;cos(x[1]);1/x[1]] #input is 1x1 array
    (_, _, _, pol) = checkDerivative(g, jacg, [.4], nothing)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1 #input is array
end

@testset "R^3 -> R^3" begin
    f = x -> [x[1]x[2]x[3]; sin(x[1]) + x[2]exp(x[3]); exp(x[2] - x[3])/x[1]]
    jacf = x -> [ x[2]x[3]  x[1]x[3] x[1]x[2];
                 cos(x[1]) exp(x[3]) x[2]exp(x[3]);
                -exp(x[2]-x[3])/x[1]^2 exp(x[2]-x[3])/x[1] -exp(x[2]-x[3])/x[1] ]
    (_, _, _, pol) = checkDerivative(f, jacf, [1;2;3], nothing)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1 #input is scalar
end





