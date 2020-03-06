include("btgDerivatives.jl")
include("kernel.jl")

using Distributions
using Printf
using SpecialFunctions
using Plots
using Polynomials
using DataFrames
using CSV
#using .btgDeriv


function normalize(X)
    for j = 1:size(X, 2)
        col = X[:, j]
        mini = minimum(col)
        maxi = maximum(col)
        if mini == maxi
            col = col/maxi
        else
            col = (col .- mini)/(maxi-mini)
        end
        X[:, j] = col
    end
    return X
end

function squash(X)
    maxi = maximum(X)
    return X ./ maxi
end


if true
    #s = [3.0; 4; 2]; s0 = [1; 2; 3]; X = [3 4; 9 5; 7 13]; X0 = [1 1; 2 1; -1 3]; z = [10; 11; 13]
    s = [3.0 6 1 3; 4 2 3 2; 2 1 1 5; 1 4 2 3;5 6 7 8]
    s0 = [1.0 2 3 2; 2 4 2 1; 3 1 2 6; 1 9 4 2; 2 3 8 6]
    X = [3.0 4 4; 9 3 5; 1 7 13; 4 1 2; 5 6 14]
    X0 = [1.0 1 2 ; 3 2 1; 4 -1 3; 5 5 4; 8 -3 5]
    z = [9.0; 11; 13; 6; 7]

    example = setting(s, s0, X, X0, z)
else 
    df = DataFrame(CSV.File("data//abalone.csv"))
    data = convert(Matrix, df[:,2:9]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight, age
    #target = convert(Matrix, df[:, 9:9]) #age

    #ind = 200:230
    ind = 220:230
    ind0 = 23:24
    block = vcat(data[ind, :], data[ind0, :]) #block contains training vectors and test vectors (values and labels)
    #s = [3 1; 4 1; 2 1; 5 6]; s0 = [1 1; 2 1; 3 1]; X = [3 4; 9 5; 7 13; 7 8]; X0 = [1 1; 2 1; -1 3]; z = [10; 11; 13;3]
    s = block[1:length(ind), 1:3]; s0 = block[length(ind)+1:end, 1:3]; X = block[1:length(ind), 1:1]; X0 = block[length(ind)+1:end, 1:1]; z = squash(block[1:length(ind), 8])
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
    θ0 = [50]
    (h, A) = checkDerivative(f, df, θ0);
    println("partial_theta function return value")
    println(polyfit(h, A, 1)) 
end

#Example 4: Check derivative of partial_theta
if true
    #f = θ -> partial_theta(θ[1], 2, example)[1](z0)
    #df = θ -> partial_theta(θ[1], 2, example)[2](z0)
    example2 = getExample(1, 25, 1, 1, 10)
    θ0 = [1.0]
    z0 = [6];
    f = θ -> partial_theta(θ[1], 2.0, example2)[2](z0)
    df = θ -> partial_theta(θ[1], 2.0, example2)[3](z0)
   
    (h, A) = checkDerivative(f, df, θ0, 5, 14, 10)
    plt1 = plot(h, A, title = "Finite Difference Derivative Checker", xlabel = "log of h", ylabel = "log of error",fontfamily=font(48, "Courier") , reuse = false)
    #plot(polyfit(h, A, 1), reuse = true)
    println("partial theta of p(z0|theta, lambda, z)")
    println(polyfit(h, A, 1))  
    nums = collect(.1:.1:20) 
    g = x -> f(x)[1]
    plt2 = plot(nums, g.(nums),xlabel = "theta", ylabel = "p(z0|theta, lambda, z)", fontfamily=font(48, "Courier") ,title = "theta vs p(z0| theta, lambda, z)")
    display(plot(plt1, plt2, fontfamily=font(48, "Courier")))
    gui()
end

#Example 5: Check derivative of Box Cox
if false
    z = 2
    f =  λ -> [boxCox(z, λ[1])]
    df = λ -> [boxCoxPrime_lambda(z, λ[1])]
    λ = [2.2323]
    (h, A) = checkDerivative(f, df, λ)
    println("Box Cox partial lambda")
    f = θ -> partial_theta(θ[1], 2, example)[1]
    df = θ -> partial_theta(θ[1], 2, example)[2]
    θ0 = [.5]
    (h, A) = checkDerivative(f, df, θ0, 4, 15, 10)
    plt1 = plot(h, A, title = "Finite Difference Derivative Checker", xlabel = "log of h", ylabel = "log of error",fontfamily=font(48, "Courier") , reuse = false)
    #plot(polyfit(h, A, 1), reuse = true)
    println("partial theta of p(z0|theta, lambda, z)")
    println(polyfit(h, A, 1)) 
    println(polyfit(h, A, 1))
end

#Example 5: Check derivative of Jacobian of Box Cox w.r.t lambda
if false
    θ = 1.312
    λ0 = [2.123]   
    z0 = rand(size(s0, 1))
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
    z0 = rand(size(s0, 1))

    f = λ -> [partial_lambda(θ, λ[1], example)[1](z0)]
    df = λ -> partial_lambda(θ, λ[1], example)[2](z0)
    (h, A) = checkDerivative(f, df, λ0)
    println("partial lambda")
    println(polyfit(h, A, 1))
end

#Example 7: Check derivative of sub-variables of posterior_theta. No "sub-functions", because we don't 
#have a z0 to deal with
if true
    println("loading some simple functions")
    @time begin
    θ0 = [2.0]
    λ = 3.0; 
    pθ = θ -> 1/sqrt(2*pi)*exp.(-(θ .- 1) .^2/2)
    dpθ = θ -> (1 .- θ)/sqrt(2*pi)*exp.(-(1 .- θ) .^2/2)
    dpθ2 = θ -> -1/(sqrt(2*pi))*exp.(-(θ .-1).^2/2) + (1 .-θ)^2 * 1/sqrt(2*pi) * exp.(-(1 .- θ)^2/2)
    pλ = λ -> 1
    end
    println("defining posterior theta and derivative")
    @time begin
    f = θ  -> posterior_theta(θ[1], λ, pθ, dpθ, dpθ2, pλ, example2)[1]
    df = θ -> posterior_theta(θ[1], λ, pθ, dpθ, dpθ2, pλ, example2)[2]
    end
    (rr, tt) = checkDerivative(θ -> [dpθ(θ[1])], θ -> dpθ2(θ[1]), [.5], 5, 12)
    println("check derivative of prior")
    println(polyfit(rr, tt, 1))
    println("Check derivative by eval f and df at 10 points")
    @time begin
    (h, A) = checkDerivative(f, df, θ0, 5, 11)
    end
    println("partial theta of p(theta, lambda|z)")
    pfit = polyfit(h, A, 1)
    println(pfit)
    println("Plotting")
    @time begin
    plt1 = plot(h, A, title = "Finite Difference Derivative Checker", xlabel = "log of h", ylabel = "log of error",
    label = [string(pfit)], fontfamily=font(48, "Courier") , reuse = false)
    end
    nums = collect(.2:.2:2)
    g = x -> f(x)[1]
    println("Evaluating f at about 25 points")
    @time begin
    gnums = g.(nums)
    end
    plt2 = plot(nums, gnums,xlabel = "theta", ylabel = "p(theta, lambda| z)", fontfamily=font(48, "Courier") ,title = "theta vs p(theta, lambda| z)")
    display(plot(plt1, plt2, fontfamily=font(48, "Courier") ))
    #display(plot(plt1))
    gui()
    
end

#Example 8: Check derivative of posterior_lambda
if false
    θ = 1; 
    pλ = λ -> 1/sqrt(2*pi)*exp.(-(λ .- 1) .^2/2)
    dpλ = θ -> (1 .- θ)/sqrt(2*pi)*exp.(-(1 .- θ) .^2/2)
    pθ = θ -> 1; 
    f = λ -> [posterior_lambda(θ, λ[1], pθ, pλ, dpλ, example)[1]]
    df = λ -> posterior_lambda(θ, λ[1], pθ, pλ, dpλ, example)[2]
    λ0 = [0.6]
    (h, A) = checkDerivative(f, df, λ0)
    println("partial lambda of p(theta, lambda|z)")
    println(polyfit(h, A, 1))
end

#Example 10: Check various derivatives in partial_z0 function
if false
    g = (x, λ) -> -exp.(λ .*x)
    dg = (x, λ) -> -λ .*exp.(λ .*x)
    dg2 = (x, λ) -> -λ^2 .*exp.(λ .*x)
   (jac, djac) =  partial_z0(2, 1, example, g, dg, dg2)
   (h, A) = checkDerivative(jac, djac, [1;20])
   println("jac and djac w.r.t z0")
   println(polyfit(h, A, 1)) 
end

#Example 11: Check derivative of rbf second derivative
if false
    x0 = [3, 4]
    f = theta -> rbf_prime(x0[1], x0[2], theta)
    df = theta -> rbf_prime2(x0[1], x0[2], theta)
    (h, A) = checkDerivative(f, df, 3.0)
    println("first and second derivatives of rbf")
    println(polyfit(h, A, 1))
end

#Example 12: Check that derivative of trace is trace of derivative
if false
    s1 = [1;2;3]
    s2 = [5;6;7]
    f = θ -> tr(K(s1, s2, θ, rbf))
    df = θ -> tr(K(s1, s2, θ, rbf_prime))

    (h, A) = checkDerivative(f, df, .5)
    println("derivative of trace")
    println(polyfit(h, A, 1))
end


