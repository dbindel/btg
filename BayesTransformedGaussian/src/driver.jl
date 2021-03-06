include("model.jl")
include("statistics.jl")
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
normalizing_constant = maximum(target)
target = target/normalizing_constant #normalization

#pick training points
#ind = 1:30
ind = 1:5
s = data[ind, :] 
#choose a subset of variables to be regressors for the mean
X = data[ind, 1:1] 
z = float(target[ind])
#normalizing_constant = maximum(z)
#z = z ./ normalizing_constant

#select range_lambda and range_theta
if false
    histogram(boxCoxObj.f.(target, .5), bins=:scott)
end
#define ranges for theta and lambda
rangeθ = [200.0, 700.0]
rangeλ = [.4, .6]

if false #look at eigenspectrum of kernel matrix
    θ = 200
    gm = K(s, s, θ, rbf) 
    display(plot(sort(eigvals(gm))))
end

#initialize train and test data structures
i = 120
s0 = data[i:i,:] #covariates and coordinates
X0  = data[i:i, 1:1]
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

if false # use this blockc to get pdf and cdf, sanity check them, and optionally plot them
    reset_timer!()
    choleskytime = 0

    #(f, g, df) = getBtgDensity(train, test, rangeθ, rangeλ, boxCoxObj, "Gaussian", "Uniform")

    i=43
    ind = [collect(1:i-1); collect(i+1:length(z))]
    train_cur = trainingData(s[ind, :], X[ind, :], z[ind]) 
    test_cur = testingData(s[i:i, :], X[i:i, :])
    (f, g, df) = getBtgDensity(train_cur, test_cur, rangeθ, rangeλ, boxCoxObj, "Gaussian", "Uniform")

    if true
    println("Plotting...")
    #plt(df, 0.1, 2, 100, "dpdf")
    plt(f, 0.1, 2, 300, "pdf")
    plt!(g, 0.1, 2, 300, "cdf")
    println(median(f, g))
    #plt(f, 0.1, 0.9, 500, "pdf")
    #plt!(g, 0.1, 0.9, 500, "cdf")
    end
end

if true #check derivative of reference function
    f = x->x[1]^2+x[2]
    df = x -> [2*x[1]; 1]
    #f = x -> x[1]^2*x[2]^2 + x[2]^4
    #df = x -> [2*x[1]*x[2]^2 ; 2*x[1]^2*x[2] + 4*x[2]^3]
    (h, A) = checkDerivative(f, df, [1.2, 2.1])
    println(polyfit(h, A, 1))
end

if true #check derivatives of CDF(z, s) w.r.t s
    function phi(s0_new)
        #println("here1")
        test.s0 = s0_new
        #println("here2")
        theta_params = funcθ(1.2, train, test, "Gaussian") 
        #println("here3")
        (r1, r2, r3, r4, r5) = partial_s(1.2, 10.5, train, test, boxCoxObj, theta_params, "Gaussian")
    end
    s0_initial = similar(test.s0)
    for i = 1: max(size(s0, 1), size(s0, 2))
        s0_initial[i] = test.s0[i] + 2*(rand()-.5)
    end
    (h, A) = checkDerivative(s0 -> phi(s0)[3], s0 -> phi(s0)[4], s0_initial, nothing,7, 20)
    println(polyfit(h, A, 1))
    plt1 = Plots.plot(h, A, title = "Finite Difference Derivative Checker", xlabel = "log of h", ylabel = "log of error",fontfamily=font(48, "Courier") , reuse = false)
    Plots.display(plt1)
    gui()
end

if false #check derivatives of df, f, or g
    (h, A) = checkDerivative(f, df, .3, nothing, 7, 15, 10)
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
    Xs, Ys = cross_validate(train, rangeθ, rangeλ, boxCoxObj, "Gaussian", "Uniform")  
    z = train.z
   # _, Xs, Ys = cross_validate(X, s, boxCox, boxCoxPrime, pθ, pλ, z, range_theta, range_lambda, 500, 2, 24)
    display(Plots.plot(Xs, Ys, 
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
    Plots.savefig("abalone_1.pdf")
end


if false #delete one group cross validation
    Xs, Ys = cross_validate_groups(X, s, boxCox, boxCoxPrime, pθ, pλ, z, rangeθ, rangeλ, 5, 500, 2, 24)
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



if false #compute loss for Abalone dataset
    graph = false #generate plots for debugging

    println("Computing loss...")
    (med,  _, _) = compute_loss(train, rangeθ, rangeλ, boxCoxObj, "Gaussian", "Uniform", graph, 100, 0, 2.5)
    println("maximum computed median: ", maximum(med))
    println("minimum computed median: ", minimum(med))
    SE = norm(med .- z)^2 * normalizing_constant^2
    println("SE: ", SE)
    
    if graph == true
        display(Plots.plot(Xs, Ys, 
        layout = length(z), 
        legend=nothing, 
        xtickfont = Plots.font(4, "Courier"),
        ytickfont = Plots.font(4, "Courier"), 
        lw=0.5))
        display(Plots.plot!(z', layout = length(z), 
        legend=nothing, 
        seriestype = :vline, 
        xtickfont = Plots.font(4, "Courier"), 
        ytickfont = Plots.font(4, "Courier"), 
        lw=0.5))
        display(Plots.plot!(med', layout = length(z), 
        legend=nothing, 
        seriestype = :vline, 
        xtickfont = Plots.font(4, "Courier"), 
        ytickfont = Plots.font(4, "Courier"), 
        lw=0.5))
    end
end

