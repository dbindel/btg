include("model.jl")
include("model_deriv.jl")
include("statistics.jl")
include("transforms.jl")
include("plotting.jl")
include("validation.jl")

using DataFrames
using CSV
#using StatsBase
using Plots
#using Profile
#using ProfileView

df = DataFrame(CSV.File("data//abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age

#pick training points
#ind = 1:30
ind = 1:30
#ind = 1:4176
s = data[ind, :] 
#X = data[ind, :] 
#X = ones(length(data))[ind]
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
range_lambda = [-3.0, 3.0]

if false #look at eigenspectrum of kernel matrix
    θ = 200
    gm = K(s, s, θ, rbf) 
    display(plot(sort(eigvals(gm))))
end

    #load examples
    i = 240
    s0 = data[i:i,:] #covariates and coordinates
    X0  = data[i:i, 1:3]
    example = setting(s, s0, X, X0, z)#abalone data
    example2 = getExample(1, 10, 1, 1, 2)

if true # use this blockcxsanity check + plot data
    @printf("sanity check and plotting block")
    pdff, cdff = model(example, boxCox, boxCoxPrime, pθ, pλ, range_theta, range_lambda)
    constant = cdff(30)
    pdfn = x -> pdff(x)/constant 
    cdfn = x -> cdff(x)/constant 
    #cdf = z0 ->  int1D(pdf, 0, z0, "3") 
    #@time begin
    #un = cdf(30)
    #end
    #cdfn = x -> cdf(x)/un #normalized cdf
    #pdfn = x -> pdf(x)/un #normalized pdf
    @time begin plt(pdfn, 0, 20) end
    #display(plot!(target[i], seriestype = :vline))
    #med = bisection(x -> cdfn(x)-0.5, 1e-3, 25, 1e-3, 10) 
    #println(med)
    reset_timer!()
    #@timeit "define pred density" ff = model_deriv(example, pθ, dpθ, dpθ2, pλ, range_theta, range_lambda)
    #@timeit "define pred density" ff = getBtgDensity(example, range_theta, range_lambda)
    ff = getBtgDensity(example, range_theta, range_lambda)
    locs = [0.0 for i = 1:1:20]
    gg = x -> ff(x)/constant
    if false
    for i = 1:1:20
        println("iteration: ", i)
        #@timeit "eval" locs[i] = gg([float(i)])
        input = [float(i)]
        println(typeof(input))
        println(input)
        locs[i] = gg(input)
    end
    print_timer()
    #@profview gg([2.0])
    plot!(1:1:20, locs)
end
end

if false #cross validation on training set
    _, Xs, Ys = cross_validate(X, s, boxCox, boxCoxPrime, pθ, pλ, z, range_theta, range_lambda, 500, 2, 24)
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
    savefig("results//abalone//abalone_cross_validation20.pdf")
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



