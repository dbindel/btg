using Test
include("../computation/finitedifference.jl")

finitediff_tests = false
finitediff_in_place_tests = true

if finitediff_tests

println("Running checkDerivative tests...")

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
end

if finitediff_in_place_tests
println("Running checkDerivative_in_place tests...")

@testset "R^1 -> R^1" begin
    store = [0.0]
    f = x -> x^2*exp(x)
    jacf = (x, store) -> (store[1] = 2x*exp(x) + x^2*exp(x); return nothing)
    (_, _, _, pol) = checkDerivative_in_place(f, jacf, .64, store, nothing) #Float64 -> Float64
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1

    f = x -> x[1]^2*exp(x[1])
    jacf = (x, store) -> (store[1] = 2x[1]*exp(x[1])+x[1]^2*exp(x[1]); return nothing)
    (_, _, _, pol) = checkDerivative_in_place(f, jacf, [.64], store, nothing) #Array{Float64, 1} -> Float64
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1
    
    #f = x -> [x[1]^2*exp(x[1])]
    #jacf = (x, store) -> (store[1] = [2x[1]*exp(x[1])+x[1]^2*exp(x[1])]; return nothing)
    #(_, _, _, pol) = checkDerivative_in_place(f, jacf, [.64], store, nothing) #Array{Float64, 1} -> Array{Float64, 1}
    #@test coeffs(pol)[end] ≈ 2 atol = 3e-1
    
    #f = x -> [x^2*exp(x)]
    #jacf = x -> [2x*exp(x) + x^2*exp(x)]
    #(_, _, _, pol) = checkDerivative(f, jacf, .64, nothing) #Float64 -> Array{Float64, 1}
    if false
    f = x-> x^2*exp(x)
    jacf = (x, store) -> (store[1] = 2x*exp(x) + x^2*exp(x); return nothing)
    hess_f = x -> 4x*exp(x) + 2*exp(x) + x^2*exp(x)
    (_, _, _, pol) = checkDerivative_in_place(f, jacf, .64, store, hess_f) #Float64 -> Float64
    @test coeffs(pol)[end] ≈ 3 atol = 3e-1
    end
end

@testset "R^3 -> R^1" begin
    store = [0.0, 0.0, 0.0]

    ff = xx ->  xx[1]^2 * sin(xx[2]*xx[3])
    jacf = (xx, store) -> (store[1] = 2*xx[1]*sin(xx[2]*xx[3]); store[2] =  xx[1]^2*xx[3]*cos(xx[2]*xx[3]); store[3] = xx[1]^2*xx[2]*cos(xx[2]*xx[3]); return nothing)
    (_, _, _, pol) = checkDerivative_in_place(ff, jacf, [.4 .5 .7], store, nothing) #initial guess has shape (3, 1)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1
    (_, _, _, pol) = checkDerivative_in_place(ff, jacf, [.4, .5, .7], store, nothing) #initial guess has shape (3,)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1
    (_, _, _, pol) = checkDerivative_in_place(ff, jacf, reshape([.4, .5, .7], 1, 3), store, nothing) #initial guess has shape (1, 3)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1

    if false
    #check Hessian   
    hess_f = x -> [2sin(x[2]x[3]) 2x[1]x[3]cos(x[2]x[3]) 2x[1]x[2]cos(x[2]x[3]) ;
                    2x[1]x[3]cos(x[2]x[3]) -x[1]^2 * x[3]^2 * sin(x[2]x[3]) x[1]^2*cos(x[2]x[3]) - x[1]^2*x[2]x[3]sin(x[2]x[3]);
                    2x[1]x[2]cos(x[2]x[3]) x[1]^2*cos(x[2]x[3]) - x[1]^2*x[2]x[3]sin(x[2]x[3]) -x[1]^2*x[2]^2 * sin(x[2]x[3])]
    (_, _, plt1, pol) = checkDerivative(ff, jacf, reshape([.4, .5, .7], 1, 3), hess_f, 2, 6) #initial guess has shape (1, 3)
    #display(plt1)
    @test coeffs(pol)[end] ≈ 3 atol = 3e-1
    end
end

@testset "R^1 -> R^3" begin
    store = [0.0, 0.0, 0.0]
    f = x -> [x;sin(x);log(x)]
    jacf = (x, store) -> (store[1] =1;store[2] = cos(x); store[3] = 1/x; return nothing)
    (_, _, _, pol) = checkDerivative_in_place(f, jacf, .4, store, nothing)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1 #input is scalar

    g = x -> [x[1];sin(x[1]);log(x[1])]
    jacg = (x, store) -> (store[1] = 1; store[2] = cos(x[1]); store[3] = 1/x[1]; return nothing) #input is 1x1 array
    (_, _, _, pol) = checkDerivative_in_place(g, jacg, [.4], store, nothing)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1 #input is array
end

@testset "R^3 -> R^3" begin
    store = zeros(3, 3)
    f = x -> [x[1]x[2]x[3]; sin(x[1]) + x[2]exp(x[3]); exp(x[2] - x[3])/x[1]]
    jacf = (x, store) -> (store[1, 1] = x[2]x[3]; store[1, 2] = x[1]x[3]; store[1, 3] = x[1]x[2];
                        store[2, 1] =  cos(x[1]); store[2, 2] = exp(x[3]); store[2, 3] = x[2]exp(x[3]);
                        store[3, 1] = -exp(x[2]-x[3])/x[1]^2; store[3, 2] = exp(x[2]-x[3])/x[1]; store[3, 3] = -exp(x[2]-x[3])/x[1]; return nothing)
    
    #[ x[2]x[3]  x[1]x[3] x[1]x[2];
    #             cos(x[1]) exp(x[3]) x[2]exp(x[3]);
    #            -exp(x[2]-x[3])/x[1]^2 exp(x[2]-x[3])/x[1] -exp(x[2]-x[3])/x[1] ]
    (_, _, _, pol) = checkDerivative_in_place(f, jacf, [1;2;3], store, nothing)
    @test coeffs(pol)[end] ≈ 2 atol = 3e-1 #input is scalar
end

end