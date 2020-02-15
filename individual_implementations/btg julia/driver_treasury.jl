using DataFrames
using CSV
using Random   
using Plots
using PDMats
using Printf
include("transforms.jl")
include("statistics.jl")
include("validation.jl")
include("plotting.jl")

function gridsearch(f, a, b, numpts=45)
    h = (b-a)/(numpts-1)
    l, m = 0, 0
    for i = a:h:b
        val = f(i); 
        if val>m 
            m = val; l = i; 
        end
    end
    return l, m
end

#macroeconomics time series data
df = DataFrame(CSV.File("data\\TB3MS.csv"))

z = convert(Matrix, df[:,2:2])
n = length(z)
s = collect(1:n) #data points equally spaced in time
portion_train = 0.3

rng = MersenneTwister(1234)
ind = shuffle(rng, s)
id_train = view(ind, 1:Int(ceil(n*portion_train)))
id_test = view(ind, (Int(ceil(n*portion_train))+1):n)

train_x = s[id_train]; test_x = s[id_test] 
train_y = z[id_train]; test_y = z[id_test]

X = ones(length(train_x), 1); X0 = [1]
pλ = x -> 1; pθ = x -> 1
range_theta = [0.1 5] 
range_lambda = [-5 5]
results = Array{Float64}(undef, length(test_x))#preallocate space for medians and interval widths
intwidths = Array{Float64}(undef, length(test_x))

if true #test and time pdfn and cdfn functions
pdf = model(X, X0, train_x, [test_x[6]], boxCox, boxCoxPrime, pθ, pλ, train_y, range_theta, range_lambda)
cdf = z0 ->  int1D(pdf, 0, z0, "0") 
@time begin
un = cdf(30)
end
cdfn = x -> cdf(x)/un #normalized cdf   
pdfn = x -> pdf(x)/un #normalized pdf
function testcdfn(x)
    @time begin
        r = cdfn(x)
    end
    return r
end

function testbisection(f, a, b)
    @time begin
        val = bisection(f, a, b)
    end
    return val
end

function testconfidence(f, med)
    @time begin
        val = confidence(f, med)
    end
    return val
end
plt2(pdfn, 0, 20)
end

if false #simply plot at multiple locations
    pltX = zeros(20, length(test_x))
    pltY = zeros(20, length(test_x))
    for i=1:length(test_x)#single-point predictions 
        println(i)
        pdf = model(X, X0, train_x, [test_x[i]], boxCox, boxCoxPrime, pθ, pλ, train_y, range_theta, range_lambda)
        cdf = z0 ->  int1D(pdf, 0, z0, "1") 
        un = cdf(30)
        cdfn = x -> cdf(x)/un #normalized cdf
        pdfn = x -> pdf(x)/un #normalized pdf
        x, y = plt(pdfn, 5, 18)
        pltX[:, i] = x; pltY[:, i] = y
    end
    display(plot(pltX, pltY, layout = length(test_x), ylim = (0, )))
end


if false #determine medians and confidence intervals  
for i=1:length(test_x)#single-point predictions 
    println(i)
    @time begin
    pdf = model(X, X0, train_x, [test_x[i]], boxCox, boxCoxPrime, pθ, pλ, train_y, range_theta, range_lambda)
    cdf = z0 ->  int1D(pdf, 0, z0, "0") 
    @time begin
    un = cdf(25)
    end
    cdfn = x -> cdf(x)/un #normalized cdf
    pdfn = x -> pdf(x)/un #normalized pdf
    @time begin
    med = bisection(x -> cdfn(x)-0.5, 1e-3, 25, 1e-2, 9) 
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
display(plot(collect(1:n), z));
display(plot!(train_x, train_y, seriestype = scatter));
display(plot!(test_x, results, seriestype = :scatter));
#display(plot!(test_x, test_y, seriestype = :scatter));
display(plot!(repeat(test_x, 2), vcat(results-intwidths, results+intwidths), seriestype = :scatter, color=:blue))
#display(plot!(test_x, test_y))
savefig("treasury6.png");
end

if true #cross validation on training set
    _, Xs, Ys = cross_validate(X, train_x, boxCox, boxCoxPrime, pθ, pλ, train_y, range_theta, range_lambda, 500, 2, 24)
    display(plot(Xs, Ys, layout = length(train_y), legend=nothing, xtickfont = font(10, "Courier"), lw=0.5))
    display(plot!(train_y', layout = length(z), legend=nothing, seriestype = :vline, xtickfont = font(10, "Courier"), lw=0.5))
    savefig("results//treasury//treasury_delete_one_cross_validate2.jpg")
end
