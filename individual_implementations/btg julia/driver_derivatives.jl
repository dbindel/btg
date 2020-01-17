include("btgDerivatives.jl")

using Distributions
using Printf
using SpecialFunctions
using Plots
using Polynomials
using DataFrames
using CSV
using .btgDeriv


"""
define inference problem using settings
s is observed prediction locations, X is matrix of covariates, z is observed values
X0 is matrix of covariates for prediction location, s0 is prediction location
"""
struct setting
    s
    s0 
    X
    X0 
    z
end

if false
    s = [3; 4; 2]; s0 = [1; 2; 3]; X = [3 4; 9 5; 7 13]; X0 = [1 1; 2 1; -1 3]; z = [10; 11; 13]
    example = setting(s, s0, X, X0, z)
else 
    df = DataFrame(CSV.File("data//abalone.csv"))
    data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
    target = convert(Matrix, df[:, 9:9]) #age

    ind = 1:15
    ind0 = 9:12
    
    #s = [3 1; 4 1; 2 1; 5 6]; s0 = [1 1; 2 1; 3 1]; X = [3 4; 9 5; 7 13; 7 8]; X0 = [1 1; 2 1; -1 3]; z = [10; 11; 13;3]
    s = data[ind, :]; s0 = data[ind0, :]; X = data[ind, 1:3]; X0 = data[ind0, 1:3]; z = target[ind]

    example = setting(s, s0, X, X0, z)
end


##Example 1
if false
    f = x -> [x.^2; x.^3]
    df = x-> [2*x;3*x] 
    x0 = [1]
    (h, A) = checkDerivative(f, df, x0)
    println(polyfit(h, A, 1))
    println("synthetic")
end

#Example 2: rbf
if false
    x = [1;2;3;4]
    f =  θ -> vec(K(x, x, θ[1], rbf));
    df = θ -> vec(K(x, x, θ[1], rbf_prime));
    θ0 = [1];
    (h, A) = checkDerivative(f, df, θ0);
    println("rbf")
    println(polyfit(h, A, 1))
end

#Example 3: Check derivatives of sub-variables of partial_theta
# Beta_hat, qtilde, Btheta, Etheta, sigmatheta, H, m, D, C, qC_inv
if false
    f = θ -> partial_theta(θ[1], 2, example)[1]  
    df = θ -> partial_theta(θ[1], 2, example)[2]
    θ0 = [3.14]
    (h, A) = checkDerivative(f, df, θ0);
    println("partial_theta function return value")
    println(polyfit(h, A, 1)) 
end

#Example 4: Check main derivative
if false
    #z0 = [1;2;3]
    z0 = rand(length(s0))
    f = θ -> [partial_theta(θ[1], 2, example)[1](z0)]  
    df = θ -> partial_theta(θ[1], 2, example)[2](z0)
    θ0 = [2.1]
    (h, A) = checkDerivative(f, df, θ0)
    println("partial_theta function return value")
    println(polyfit(h, A, 1)) 
end

#Example 5: Check derivative of Box Cox
if false
    z = 2
    f =  λ -> [boxCox(z, λ[1])]
    df = λ -> [boxCoxPrime_lambda(z, λ[1])]
    λ = [2.2323]
    (h, A) = checkDerivative(f, df, λ)
    println("Box Cox partial lambda")
    println(polyfit(h, A, 1))
end

#Example 5: Check derivative of Jacobian of Box Cox w.r.t lambda
if false
    θ = 1.312
    λ0 = [2.123]   
    z0 = rand(length(s0))
    f = λ -> [partial_lambda(θ, λ[1], example)[1](z0)]
    df = λ -> [partial_lambda(θ, λ[1], example)[2](z0)]
    (h, A) = checkDerivative(f, df, λ0)
    println("Jacobian w.r.t lambda")
    println(polyfit(h, A, 1))
end

#Example 6: Check derivative of sub-variables of partial_lambda
if false
    θ = 1.312
    λ0 = [2.123]   
    f = λ -> [partial_lambda(θ, λ[1], example)[1]]
    df = λ -> [partial_lambda(θ, λ[1], example)[2]]
    (h, A) = checkDerivative(f, df, λ0)
    println("partial lambda")
    println(polyfit(h, A, 1))
end

#Example 7: Check derivative of sub-functions of partial_lambda
if false
    θ = 1
    λ0 = [.5]  
    #z0 = [10;11;13]
    z0 = rand(length(s0))

    f = λ -> [partial_lambda(θ, λ[1], example)[1](z0)]
    df = λ -> partial_lambda(θ, λ[1], example)[2](z0)
    (h, A) = checkDerivative(f, df, λ0)
    println("partial lambda")
    println(polyfit(h, A, 1))
end

#Example 7: Check derivative of sub-variables of posterior_theta. No "sub-functions", because we don't 
#have a z0 to deal with
if true
    θ0 = [.7]
    λ = .4; 
    pθ = θ -> 1/sqrt(2*pi)*exp.(-(θ .- 1) .^2/2); pλ = λ -> 1; dpθ = θ -> (1 .- θ)/sqrt(2*pi)*exp.(-(1 .- θ) .^2/2)
    f = θ  -> vec(posterior_theta(θ[1], λ, pθ, dpθ, pλ, example)[1])
    df = θ -> vec(posterior_theta(θ[1], λ, pθ, dpθ, pλ, example)[2])
    
    (rr, tt) = checkDerivative(θ -> [pθ(θ[1])], θ -> dpθ(θ[1]), [.5])
    println(polyfit(rr, tt, 1))

    (h, A) = checkDerivative(f, df, θ0)
    println("partial lambda")
    println(polyfit(h, A, 1))
end
