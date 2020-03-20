include("model.jl")
include("model_deriv.jl")
include("statistics.jl")
include("transforms.jl")
include("plotting.jl")
include("validate.jl")
using DataFrames
using CSV
#using StatsBase
using Plots
#using Profile
#using ProfileView

df = DataFrame(CSV.File("data//abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age
target = target/maximum(target) #normalization

#pick training points
#ind = 1:30
ind = 1:50
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

if true # use this blockcxsanity check + plot data
    
    reset_timer!()
    choleskytime = 0

    (f, g) = getBtgDensity(train, test, range_theta, range_lambda, boxCoxObj, "Gaussian")

    plt(f, 0.1, 2, 100, "pdf")
    plt!(g, 0.1, 2, 100, "cdf")

end

if true #cross validation on training set
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



