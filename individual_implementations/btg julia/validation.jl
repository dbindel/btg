include("plotting.jl")

"""
Structure for storing summary statistics
"""
struct sts 
    median::Float64
    intwidth::Float64
end

"""
Single point cross-validation. Currently does not use confidence intervals or median finding. 
"""
function cross_validate(X, s, g, gprime, pθ, pλ, z, rangeθ, rangeλ, num_pts=200, strt = 0, endpt=20)
    ss = Array{sts}(undef, length(z))
    x = collect(range(strt, stop = endpt, length = num_pts)) #define mesh 
    Xs = repeat(x, outer = [1, length(z)]) #preallocate space to store data
    Ys = Array{Float64}(undef, num_pts, length(z))
    for i=1:length(z)
        println(i)
        ind = [collect(1:i-1); collect(i+1:length(z))]
        @time begin
        X0 = size(X, 2)==1 ? X[1:1, :] : X[i:i, :] # set design matrix 
        pdf = model(X[ind, :], X0, s[ind, :], s[i:i, :], boxCox, boxCoxPrime, pθ, pλ, z[ind], range_theta, range_lambda) 
        cdf = z0 ->  int1D(pdf, 0, z0, "3"); un = cdf(25) #compute unnormalized cdf
        cdfn = x -> cdf(x)/un; pdfn = x -> pdf(x)/un  #compute normalized pdf and cdf
        end
        #@time begin
        #med = bisection(x -> cdfn(x)-0.5, 1e-3, 50, 1e-3, 15) #compute median 
        #end
        #@time begin
        #intwidth = confidence(pdfn, med) #compute width of credible interval corresponding to 95% confidence level
        #end
        #st = sts(med, intwidth); ss[i]=st
        for j = 1:num_pts
            Ys[j, i] = pdfn(x[j])
        end
    end
    return ss, Xs, Ys
end