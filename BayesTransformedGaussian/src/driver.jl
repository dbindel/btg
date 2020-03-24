include("model.jl")
#include("statistics.jl")
include("transforms.jl")
include("tools/plotting.jl")
include("validation/validate.jl")

#used for testing derivatives 
include("computation/derivatives.jl")
include("computation/buffers.jl")
using DataFrames
using CSV
#using StatsBase
using Plots
#using Profile
#using ProfileView

df = DataFrame(CSV.File("datasets/abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age
target = target/maximum(target) #normalization

#pick training points
#ind = 1:30
ind = 1:40
s = data[ind, :] 
#choose a subset of variables to be regressors for the mean
X = data[ind, 1:3] 
z = float(target[ind])
#prior marginals are assumed constant
pλ = x -> 1 
pθ = x -> 1
dpθ = x -> 0 
dpθ2 = x -> 0

#define ranges for theta and lambda
range_theta = [100.0, 300.0]
range_lambda = [-1.0, 1.0]

if false #look at eigenspectrum of kernel matrix
    θ = 200
    gm = K(s, s, θ, rbf) 
    display(plot(sort(eigvals(gm))))
end

    #load examples
    i = 240
    s0 = data[i:i,:] #covariates and coordinates
    X0  = data[i:i, 1:3]
    #example = setting(s, s0, X, X0, z)#abalone data
    train = trainingData(s, X, z)
    test = testingData(s0, X0)
    #example2 = getExample(1, 10, 1, 1, 2)
    println("Data loaded.")

#experiment to see how large n has to be for Cholesky to have the most expensive computational cost
if false
    data  = repeat(data, 10, 1)
    times2 = zeros(2, 4)
    for i = 10000:10000:40000
        println("iteration: ", i/10000)
        ind = 1:i
        s = data[ind, :] 
        tK = @time K = fastK(s, s, 1.2)
        tC = @time cholesky(K)
        times[1, Int64(i/10000)] = tK;
        times[2, Int64(i/10000)] = tC;
    end    
end

  #check derivative of p(z0|theta, lambda, z)
if false
    theta_params = funcθ(1.5, train, test, "Gaussian")
    (f, df) = partial_z0(1.5, 2.0, train, test, boxCoxObj, theta_params, "Gaussian") 
    (h, A) = checkDerivative(f, df, .6, 9, 15, 10)
    plt1 = Plots.plot(h, A, title = "Finite Difference Derivative Checker", xlabel = "log of h", ylabel = "log of error",fontfamily=font(48, "Courier") , reuse = false)
    #plot(polyfit(h, A, 1), reuse = true)
    println("partial theta of p(z0|theta, lambda, z)")
    println(polyfit(h, A, 1))  
    nums = collect(.05:.05:1) 
    g = x -> f(x)[1]
    plt2 = Plots.plot(nums, g.(nums),xlabel = "theta", ylabel = "p(z0|theta, lambda, z)", fontfamily=font(48, "Courier") ,title = "theta vs p(z0| theta, lambda, z)")
    display(Plots.plot(plt1, plt2, fontfamily=font(48, "CoSurier")))
    gui()
end

if true # use this blockcxsanity check + plot data
    reset_timer!()
    choleskytime = 0

    (f, g, df) = getBtgDensity(train, test, range_theta, range_lambda, boxCoxObj, "Gaussian", "Uniform")

    println("Plotting...")
    plt(df, 0.1, 2, 100, "dpdf")
    plt!(f, 0.1, 2, 100, "pdf")
    plt!(g, 0.1, 2, 100, "cdf")

    #plt(f, 0.1, 0.9, 500, "pdf")
    #plt!(g, 0.1, 0.9, 500, "cdf")

end

#check derivatives of df, f, or g
if true
    (h, A) = checkDerivative(f, df, .3, 7, 15, 10)
    plt1 = Plots.plot(h, A, title = "Finite Difference Derivative Checker", xlabel = "log of h", ylabel = "log of error",fontfamily=font(48, "Courier") , reuse = false)
    #plot(polyfit(h, A, 1), reuse = true)
    println("derivative of p(z0|z)")
    println(polyfit(h, A, 1))  
    nums = collect(.05:.05:1) 
    gg = x -> f(x)[1]
    plt2 = Plots.plot(nums, gg.(nums),xlabel = "theta", ylabel = "p(z0|theta, lambda, z)", fontfamily=font(48, "Courier") ,title = "theta vs p(z0| theta, lambda, z)")
    display(Plots.plot(plt1, plt2, fontfamily=font(48, "CoSurier")))
    gui()
end


if false #cross validation on training set
    Xs, Ys = cross_validate(train, range_theta, range_lambda, boxCoxObj, "Gaussian", "Uniform")  
    z = train.z
   # _, Xs, Ys = cross_validate(X, s, boxCox, boxCoxPrime, pθ, pλ, z, range_theta, range_lambda, 500, 2, 24)
    display(plot(Xs, Ys, 
    layout = length(z), 
    legend=nothing, 
    xtickfont = Plots.font(4, "Courier"),
    ytickfont = Plots.font(4, "Courier"), 
    lw=0.5))
    display(plot!(z', layout = length(z), 
    legend=nothing, 
    seriestype = :vline, 
    xtickfont = Plots.font(4, "Courier"), 
    ytickfont = Plots.font(4, "Courier"), 
    lw=0.5))
    savefig("abalone_1.pdf")
end


if false #delete one group cross validation
    Xs, Ys = cross_validate_groups(X, s, boxCox, boxCoxPrime, pθ, pλ, z, range_theta, range_lambda, 5, 500, 2, 24)
    display(plot(Xs, Ys, 
    layout = length(z), 
    legend=nothing, 
    xtickfont = Plots.font(4, "Courier"),
    ytickfont = Plots.font(4, "Courier"), 
    lw=0.5))
    display(plot!(z', layout = length(z), 
    legend=nothing, 
    seriestype = :vline, 
    xtickfont = Plots.font(4, "Courier"), 
    ytickfont = Plots.font(4, "Courier"), 
    lw=0.5))
    savefig("results//abalone//abalone_group_cross_validation4.pdf")
end



