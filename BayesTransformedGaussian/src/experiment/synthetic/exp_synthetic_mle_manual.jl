using Dates
using ArgParse
using Printf
using Random
using GaussianProcesses

# before_all = Dates.now()

include("../../btg.jl")

###
###  Manually enter parsed args entries so we can run this file as a script
###
##
parsed_args = Dict()
push!(parsed_args, "randseed" => 1234)
push!(parsed_args, "lmax" => 2)
push!(parsed_args, "lmin" => 0.2)
push!(parsed_args, "noiselevel" => 1)
push!(parsed_args, "ntest" => 50)
push!(parsed_args, "ntrain" => 8)
push!(parsed_args, "p" => 1)

function covariate_fun(x, p)
    n = size(x, 1)
    d = size(x, 2)
    if p == 1
        return ones(n, 1)
    elseif p == 1 + d
        return hcat(ones(n), x)
    else
        throw(ArgumentError("Only support constant or linear convariate."))
    end
end

true_fun(x, s) = sin(x) + s*randn() + 3
true_fun_noise(x) = max(true_fun(x, 1e-1), 0)^1/3
true_fun_0(x) = true_fun(x, 0.)^1/3

# training set
randseed = parsed_args["randseed"]; rng = MersenneTwister(randseed)
n_train = parsed_args["ntrain"]
data = range(-pi, stop=pi, length=n_train)
# data = 2pi .* rand(n_train) .- pi
target = (sin.(data) .+ 0.1 * parsed_args["noiselevel"].* randn(rng, n_train) .+ 3).^(1/3)
x = reshape(data, n_train, 1)
Fx = covariate_fun(x, parsed_args["p"])
max_train = maximum(target)
y = target ./ max_train
trainingData0 = trainingData(x, Fx, y) 
d = getDimension(trainingData0); n = getNumPts(trainingData0); p = getCovDimension(trainingData0)
# testing set 
n_test = parsed_args["ntest"]
x_test = range(-pi, stop=pi, length=n_test)
x_test = reshape(x_test, n_test, 1)
Fx_test = covariate_fun(x_test, parsed_args["p"])
y_test_true = (sin.(x_test) .+ 3).^(1/3)
elapsedmin = 0

#parameter setting
# myquadtype = parsed_args["sparse"] ? ["SparseCarlo", "SparseCarlo"] : ["QuasiMonteCarlo", "QuasiMonteCarlo"]
myquadtype = ["Gaussian", "Gaussian"]
#rangeλ = [-1.5 1.] 
rangeλ = reshape([0.0], 1, 1)
lmin = parsed_args["lmin"]
lmax = parsed_args["lmax"]
#rangeθ = [1/lmax^2 1/lmin^2]
rangeθ = reshape([5.0], 1, 1)
@info "rangeθ, rangel:" rangeθ, [lmin lmax]
# rangeθ = [0.111 25]   
# build btg model
btg0 = btg(trainingData0, rangeθ, rangeλ; quadtype = myquadtype)
(pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw) = solve(btg0);

####################################
######## Optional Plotting #########
####################################
if false
plt(y -> max_train*fzero(x -> cdf0_raw(reshape([y], 1, 1), reshape([1.0], 1, 1), x) - 0.5, 0, 10), -3, 3)
end

#print(btg0.train_buffer_dict[0.5])
print(btg0.θλbuffer_dict[(5.0, 0.0)])
print(btg0.λbuffer_dict[0.0])
#print(btg0.trainingData)

####################################
############### Test ###############
####################################
if true
@info "Start Test"
before = Dates.now()
count_test = 0
error_abs = 0.
error_sq = 0.
nlpd = 0.
id_fail = []
id_nonproper = []
CI_set = zeros(n_test, 2)
median_set = zeros(n_test)
for i in 1:n_test
    global error_abs, error_sq, nlpd, count_test
    # mod(i, 20) == 0 ? (@info i) : nothing
    # @info "i" i
    x_test_i = reshape(x_test[i, :], 1, d)
    Fx_test_i = reshape(Fx_test[i, :], 1, p)
    try
        pdf_test_i, cdf_test_i, dpdf_test_i, quantbound_test_i, support_test_i = pre_process(x_test_i, Fx_test_i, pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw)
        y_test_i_true = y_test_true[i]
        median_test_i = max_train * quantile(cdf_test_i, quantbound_test_i, support_test_i)[1]
        # @info "True, median " y_test_i_true, median_test_i
        median_set[i] = median_test_i
        try 
            CI_test_i = max_train .* credible_interval(cdf_test_i, quantbound_test_i, support_test_i; mode=:equal, wp=.95)[1]
            count_test += (y_test_i_true >= CI_test_i[1])&&(y_test_i_true <= CI_test_i[2]) ? 1 : 0
            # @info "CI" CI_test_i
            @info CI_test_i
            CI_set[i, :] = CI_test_i
        catch err
            append!(id_fail, i)
        end
        error_abs += abs(y_test_i_true - median_test_i)
        error_sq += (y_test_i_true - median_test_i)^2
        nlpd += log(pdf_test_i(y_test_i_true/max_train)) 
    # @info "Count, id_fail" count_test, id_fail
    catch err 
        append!(id_nonproper, i)
    end
    count_test /= n_test - length(id_fail) - length(id_nonproper)
    error_abs  /= n_test - length(id_nonproper)
    error_sq   /= n_test - length(id_nonproper)
    nlpd       /= -n_test - length(id_nonproper)
    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)

end
    # Plot
    PyPlot.close("all") #close existing windows
    PyPlot.plot(x_test, median_set, label = "BTG median")
    PyPlot.plot(x_test, y_test_true, label = "true")
    PyPlot.fill_between(dropdims(x_test; dims = 2), CI_set[:, 1], CI_set[:, 2], alpha = 0.3, label = "95% confidence interval")
    PyPlot.scatter(x, target, s = 10, c = "k", marker = "*")
    PyPlot.legend(fontsize=8)
    PyPlot.grid()
    PyPlot.title("BTG $myquadtype", fontsize=10)
    PyPlot.savefig("exp_synthetic_mle_btg4.pdf")
end