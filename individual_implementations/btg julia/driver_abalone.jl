include("model.jl")
include("statistics.jl")
include("transforms.jl")
include("plotting.jl")
include("validation.jl")

using DataFrames
using CSV
using StatsBase
using Plots

df = DataFrame(CSV.File("data//abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age

#pick training points
ind = 1:40
s = data[ind, :] 
#X = data[ind, :] 
#X = ones(length(data))[ind]
#choose a subset of variables to be regressors for the mean
X = data[ind, 1:3] 
z = target[ind]

#prior marginals are assumed constant
pλ = x -> 1 
pθ = x -> 1

#define ranges for theta and lambda
range_theta = [100 300]
range_lambda = [-3 3]

if false # use this blockcxsanity check + plot data
    @printf("hi")
    i = 240
    s0 = data[i:i,:] #covariates and coordinates
    X0  = data[i:i, 1:3]
    pdff, cdff = model(X, X0, s, s0, boxCox, boxCoxPrime, pθ, pλ, z, range_theta, range_lambda)
    constant = cdff(30)
    pdfn = x -> pdff(x)/constant
    cdfn = x -> cdff(x)/constant 
    #cdf = z0 ->  int1D(pdf, 0, z0, "3") 
    #@time begin
    #un = cdf(30)
    #end
    #cdfn = x -> cdf(x)/un #normalized cdf
    #pdfn = x -> pdf(x)/un #normalized pdf
    @time begin plt(pdfn, 0, 30) end
    #display(plot!(target[i], seriestype = :vline))
    #med = bisection(x -> cdfn(x)-0.5, 1e-3, 25, 1e-3, 10) 
    #println(med)
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
    savefig("results//abalone//abalone_cross_validation13.pdf")
end


if true #delete one group cross validation
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



