include("plotting.jl")

"""
Structure for storing summary statistics
"""
struct sts 
    median::Float64
    intwidth::Float64
end

function cross_validate(train::trainingData{A, B}, rangeθ::B, rangeλ::B, transform, quadtype = "Gaussian", priortype = "Uniform", num_pts=100, strt = 0, endpt=1.5) where A<:Array{Float64, 2} where B<:Array{Float64, 1}
    X = train.X
    x = collect(range(strt, stop = endpt, length = num_pts)) #define mesh 
    Xs = repeat(x, outer = [1, length(z)]) #preallocate space 
    Ys = Array{Float64}(undef, num_pts, length(z))
    for i=1:length(z)
        println(i)
        ind = [collect(1:i-1); collect(i+1:length(z))]
        @time begin
        train_cur = trainingData(s[ind, :], X[ind, :], z[ind]) 
        test_cur = testingData(s[i:i, :], X[i:i, :])
        pdf, cdf = getBtgDensity(train_cur, test_cur, rangeθ, rangeλ, transform, quadtype, priortype)
        end
        for j = 1:num_pts
            Ys[j, i] = pdf(x[j])
        end
    end
        return Xs, Ys
end

"""
Single point cross-validation. Currently does not use confidence intervals or median finding. 
"""
function cross_validate_artifact(X, s, g, gprime, pθ, pλ, z, rangeθ, rangeλ, num_pts=200, strt = 0, endpt=20)
    ss = Array{sts}(undef, length(z))
    x = collect(range(strt, stop = endpt, length = num_pts)) #define mesh 
    Xs = repeat(x, outer = [1, length(z)]) #preallocate space 
    Ys = Array{Float64}(undef, num_pts, length(z))
    for i=1:length(z)
        println(i)
        ind = [collect(1:i-1); collect(i+1:length(z))]
        @time begin
        X0 = size(X, 2)==1 ? X[1:1, :] : X[i:i, :] # constant mean?
        pdf, cdf = model(X[ind, :], X0, s[ind, :], s[i:i, :], boxCox, boxCoxPrime, pθ, pλ, z[ind], range_theta, range_lambda) #X0 is [1]
        constant = cdf(30)
        pdfn = x -> pdf(x)/constant
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

"""
Delete-one-group cross validation.
"""
function cross_validate_groups(X, s, g, gprime, pθ, pλ, z, rangeθ, rangeλ, num_group = 5, num_pts=200, strt = 0, endpt=20)
    if num_group>length(z) 
        error("number of splits must be less than or equal to number of data points")
    end
    x = collect(range(strt, stop = endpt, length = num_pts)) #define mesh 
    Xs = repeat(x, outer = [1, length(z)]) #preallocate space to store 
    Ys = Array{Float64}(undef, num_pts, length(z)) 
    num_per_group = Integer(ceil(length(z)/num_group))
    for i=1:num_group
        ind_train = [collect(1:(i-1)*num_per_group); collect(i*num_per_group+1:length(z))] #delete the ith group   
        println(ind_train)  
        for k=1:num_per_group    
            cur = (i-1)*num_per_group + k
            println(cur)
            if cur <= length(z)
                X0 = size(X, 2)==1 ? X[1:1, :] : X[cur:cur, :] # set design matrix 
                @time begin
                pdf, cdf = model(X[ind_train, :], X0, s[ind_train, :], s[cur:cur, :], boxCox, boxCoxPrime, pθ, pλ, z[ind_train], range_theta, range_lambda) 
                constant = cdf(30) #compute unnormalized cdf
                cdfn = x -> cdf(x)/constant; pdfn = x -> pdf(x)/constant  #compute normalized pdf and cdf
                end
                for j = 1:num_pts
                    Ys[j, cur] = pdfn(x[j])
                end
            end
        end
    end
    return Xs, Ys
end