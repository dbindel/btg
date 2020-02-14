include("transforms.jl")
include("kernel.jl")
include("model.jl")
include("prediction.jl")
include("statistics.jl")
using Plots
using Random 
using Printf

Random.seed!(1234);
s=[-10 1 2 10 12 24 30 56 80 85 90 100]
s0 = [5]
K1 = K(s, s, 12, rbf)
z = sample(K1) #zero mean data with fixed covariance structure
mean = (minimum(z) < 0 ? abs(minimum(z)) + 12 : 12.1); 
z = z .+  mean; # sample positive data with fixed mean 
z = map(x -> invBoxCox(x, 2), z) 
println(z) 
X = ones(length(s), 1) #assume constant mean b/c of lack of covariate information
X0 = ones(length(s0), 1)
pλ = x -> 1 #let lambda be uniformly distributed (value of constant is immaterial, because of renormalization) 
pθ = x -> 1 #let theta be uniformly distributed  

if false #DEPRECATED
    pdf = z0 -> prediction(X, X0, s, s0, boxCox, boxCoxPrime, pθ, pλ, z, [z0], [10 14], [1 3])
    cdf = z0 ->  int1D(pdf, 0, z0) 
    un = cdf(100)
    @printf("Unnormalized Integral of Bayesian Predictive Density Function: %f\n", un) 
    cdfn = x -> cdf(x)/un #normalized cdf
    pdfn = x -> pdf(x)/un #normalized pdf
    @time begin
    med = bisection(x -> cdfn(x)-0.5, 1e-3, 12, 1e-3, 120) 
    end
    @printf("Median of CDF: %f \n", med)
    x = range(.01, stop=20, length= 120)
    y= []
    for i = 1:120
        append!(y, prediction(X, X0, s, s0, boxCox, boxCoxPrime, pθ, pλ, z, [x[i]], [12 14], [1 3]))
    end
    display(plot(x, y, seriestype = :line))
    display(plot!([med], seriestype = :vline))

    @time begin
    h = confidence(pdfn, med) 
    end
    @printf("interval width: %f \n: ", h)
    display(plot!([med-h, med+h], seriestype = :vline))
end
"""
Structure for storing statistics
"""
struct sts 
    median::Float64
    intwidth::Float64
end 
"""
Use single point deletion to validate model
"""
function cross_validate(X, X0, s, g, gprime, pθ, pλ, z, rangeθ, rangeλ)
    ss = Array{sts}(undef, length(z))
    num_pts = 400 #number of points used in plots 
    endpt = 20 #right end-point of plot for the pdf 
    x = collect(range(.001, stop = endpt, length = num_pts))
    X = repeat(x, outer = [1, length(z)]) #preallocate space for plotting data
    Y = Array{Float64}(undef, num_pts, length(z))
    for i=1:length(z)
        println(i)
        ind = [collect(1:i-1); collect(i+1:length(z))]
        @time begin
        pdf = z0 -> prediction(X[ind], [X[i]], s[ind], [s[i]], g, gprime, pθ, pλ, z[ind], [z0], rangeθ, rangeλ)#leave out one index
        cdf = z0 ->  int1D(pdf, 0, z0); un = cdf(50) #compute unnormalized cdf
        cdfn = x -> cdf(x)/un; pdfn = x -> pdf(x)/un  #compute normalized pdf and cdf
        end
        @time begin
        med = bisection(x -> cdfn(x)-0.5, 1e-3, 50, 1e-3, 15) #compute median 
        end
        @time begin
        intwidth = confidence(pdfn, med) #compute width of credible interval corresponding to 95% confidence level
        end
        st = sts(med, intwidth); ss[i]=st
        @time begin
        for j = 1:num_pts
            Y[j, i] = pdfn(x[j])
        end
        end
    end
    return ss, X, Y
end

if true
    svals, X, Y = cross_validate(X, X0, s, boxCox, boxCoxPrime, pθ, pλ, z, [11.5 12.5], [1.95 2.05])
    ss = Array{Float64}(undef, 3, length(s))
    for j = 1:length(s)
        ss[1, j] = svals[j].median-svals[j].intwidth
        ss[2, j] = svals[j].median 
        ss[3, j] = svals[j].median+svals[j].intwidth
    end
    display(plot(X, Y, layout=length(s), legend = nothing, ylims = (0, maximum(Y)+.03)))
    display(plot!(ss, seriestype = :vline, layout = length(s)))
    display(plot!(z', seriestype = :vline, layout=length(s)))
    savefig("validate18.png")
end
