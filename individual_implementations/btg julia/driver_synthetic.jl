include("transforms.jl")
include("kernel.jl")
include("model.jl")
include("prediction.jl")
include("statistics.jl")
include("validation.jl")
using Plots
using Random 
using Printf

function plt(f, a, b, numpts=500)
    h = (b-a)/(numpts-1)
    y = zeros(1,numpts)
    x = collect(a:h:b)
    y = map(f, x)
    display(plot(x, y))
end

#Random.seed!(1234); 
Random.seed!(4312)
h = 50
smaster=collect(0:1/h:1-1/h)
s0 = [4]
K1 = K(smaster, smaster, 120, rbf)
zo = sample(K1) #zero mean data with fixed covariance structure
mean = (minimum(zo) < 0 ? abs(minimum(zo)) + 12 : 12.1); 
zo = zo .+  mean; # sample positive data with fixed mean 
zmaster = map(x -> invBoxCox(x, 2), zo) 
pλ = x -> 1 #let lambda be uniformly distributed (value of constant is immaterial, because of renormalization) 
pθ = x -> 1 #let theta be uniformly distributed  
#display(plot(repeat(smaster, 1, 2), hcat(zo, z), seriestype = :scatter, layout=2, title = "boxcox"))
ind_master = shuffle(collect(1:h))
ind = ind_master[1:15]
s = (collect(0:1/h:1-1/h))[ind]; z  = zmaster[ind]
display(plot(smaster, zmaster, seriestype = :scatter))
display(plot!(s, z, seriestype = :scatter))
#savefig("masters.png")
#println(s)
range_theta = [50 250] 
range_lambda = [-3 6]
X = ones(length(s), 1) #assume constant mean b/c of lack of covariate information
X0 = ones(1, 1)

totest = [30 31 32 33 34 35 36] #selected testing location indices (of smaster)
#totest = [11 12 13 14 15]
results = Array{Float64}(undef, length(totest)) #preallocate space for medians and interval widths
intwidths = Array{Float64}(undef, length(totest))

if false #sanity check + plot data
    pdf = model(X, X0, s, [smaster[totest[3]]], boxCox, boxCoxPrime, pθ, pλ, z, range_theta, range_lambda)
    cdf = z0 ->  int1D(pdf, 0, z0, "3") 
    @time begin
        un = cdf(23)
    end
    cdfn = x -> cdf(x)/un #normalized cdf
    pdfn = x -> pdf(x)/un #normalized pdf
    plt(pdfn, 0, 23)
    savefig("results//synthetic//data3.jpg")
    med = bisection(x -> cdfn(x)-0.5, 1e-3, 25, 1e-3, 10) 
    println(med)
end

if false #test prediction at multiple locations
    for i=1:length(totest) #single-point predictions 
        println(i)
        @time begin
        pdf = model(X, X0, s, [smaster[totest[i]]], boxCox, boxCoxPrime, pθ, pλ, z, range_theta, range_lambda)
        cdf = z0 ->  int1D(pdf, 0, z0, "3") 
        @time begin
        un = cdf(23)
        end
        cdfn = x -> cdf(x)/un #normalized cdf
        pdfn = x -> pdf(x)/un #normalized pdf
        @time begin
        med = bisection(x -> cdfn(x)-0.5, 1e-3, 25, 1e-3, 10) 
        #med, val = gridsearch(pdfn, 0, 30, 20) #MAP estimation
        @printf("ith pt med: %f \n", med)
        end
        @time begin
        intwidths[i] = confidence(pdfn, med) #compute width of credible interval corresponding to 95% confidence level
        @printf("ith pt intwidth: %f \n", intwidths[i])
        end
        results[i] = med
        end
    end
display(plot(smaster, zmaster, seriestype = :scatter, label = "latent process", title = "\\theta [50, 150], \\lambda [1, 3], actual (\\theta = 120, \\lambda = 5)"));
display(plot!(s, z, seriestype = :scatter, label = "training"));
display(plot!(smaster[totest'], results, seriestype = :scatter, label = "predicted median"));
#display(plot!(test_x, test_y, seriestype = :scatter));
display(plot!(repeat(smaster[totest]', 2), vcat(results-intwidths, results+intwidths), seriestype = :scatter, color=:red, label = "95% confidence"));
#display(plot!(test_x, test_y))
savefig("synthetic7.png")
end

if true #cross validation
    ss, Xs, Ys = cross_validate(X, s, boxCox, boxCoxPrime, pθ, pλ, z, range_theta, range_lambda, 1000, 2, 20)
    display(plot(Xs, Ys, layout = length(z), legend=nothing, xtickfont = font(10, "Courier"), lw=0.5))
    display(plot!(z', layout = length(z), legend=nothing, seriestype = :vline, xtickfont = font(10, "Courier"), lw=0.5))
    savefig("results//synthetic//synthetic_delete_one_cross_validate4.jpg")
end


