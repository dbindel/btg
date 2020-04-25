using Dates
using ArgParse
using Printf
using Random

include("../btg.jl")
include("../datasets/load_abalone.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "--validate"
        help = "do cross validation or not"
        action = :store_true
    "--fast"
        help = "use fast or not"
        action = :store_true
end
parsed_args = parse_args(ARGS, s)

# shuffle data
ind_shuffle = randperm(MersenneTwister(1234), size(data, 1)) 
data = data[ind_shuffle, :]
target = target[ind_shuffle]
id_train = 1:200; posx = 1:7; posc = 1:3
x = data[id_train, posx] 
Fx = data[id_train, posc] 
y = float(target[id_train])
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)

#parameter setting
rangeθ = [10.0 1000]
rangeθm = repeat(rangeθ, d, 1)
rangeλ = [-1. 1.] #we will always used 1 range scale for lambda
myquadtype = ["SparseGrid", "Gaussian"]
btg1 = btg(trainingData1, rangeθm, rangeλ; quadtype = myquadtype)
(pdf, cdf, dpdf, quantInfo) = solve(btg1); #initialize training_buffer_dicts, solve once so can use fast techiques to extrapolate submatrix determinants, etc.

####################################
############### Test ###############
####################################
if filesize("Exp_abalone_test.txt") == 0
    # write the setting headers
    io1 = open("Exp_abalone_test.txt", "w") 
    write(io1, "Time     ;    ind_train  ;  ind_test   ;      quadtype     ;     rangeθ    ;    rangeλ   ;  error_abs ;  error_sq ;  CI accuracy ; -log(p(y))  ;  failed id \n")
    close(io1)
end

id_test = 1001:1200
n_test = length(id_test)
id_fail = []
Fx0 = reshape(data[id_test, posc], n_test, length(posc))
x0 = reshape(data[id_test, posx], n_test, length(posx)) 
count_test = 0
error_abs  = 0.
error_sq   = 0.
nlpd       = 0.
for i in 1:n_test
    x0_i = reshape(x0[i, :], 1, length(posx))
    Fx0_i = reshape(Fx0[i, :], 1, length(posc))
    pdf_i, cdf_i, dpdf_i, quantbound_i, support_i, fail_i = pre_process(x0_i, Fx0_i, pdf, cdf, dpdf, quantInfo)
    y_i_true = getLabel(btg1.trainingData)[i]
    median_i = quantile(cdf_i, quantbound_i, support_i)[1]
    try 
        CI_i = credible_interval(cdf_i, quantbound_i, support_i; mode=:equal, wp=.95)[1]
        count_test += (y_i_true >= CI_i[1])&&(y_i_true <= CI_i[2]) ? 1 : 0
    catch err
        append!(id_fail, i)
    end
    error_abs += abs(y_i_true - median_i)
    error_sq += (y_i_true - median_i)^2
    nlpd += log(pdf_i(y_i_true)) 
end
count_test /= n_test - length(id_fail)
error_abs  /= n_test
error_sq   /= n_test
nlpd       /= -n_test

io1 = open("Exp_abalone_test.txt", "a") 
write(io1, "$(Dates.now())  ;    $ind    ;   $id_test   ;     $myquadtype     ;   $rangeθ   ;  $rangeλ ;   $count_test    ;  $error_abs   ;  $error_sq   ;   $nlpd   ;  $id_fail  \n")
close(io1)


####################################
############ Validation ############
####################################
if parsed_args["validate"]
    if filesize("Exp_abalone_validate.txt") == 0
        # write the setting headers
        io2 = open("Exp_abalone_validate.txt", "w") 
        write(io2, "Time     ;    quadtype    ;     rangeθ    ;    rangeλ   ;  fast ;  elapsedmin ; CI accuracy\n")
        close(io2)
    end
    function lootd(td::AbstractTrainingData, i::Int64)
        x = getPosition(td)
        Fx = getCovariates(td)
        z = getLabel(td)
        x_minus_i = x[[1:i-1;i+1:end], :]
        Fx_minus_i = Fx[[1:i-1;i+1:end], :]
        z_minus_i = z[[1:i-1;i+1:end]]
        x_i = x[i:i, :]
        Fx_i = Fx[i:i, :]
        z_i = z[i:i, :]
        return trainingData(x_minus_i, Fx_minus_i, z_minus_i), x_i, Fx_i, z_i
    end

    # select random 48 points in trainig set to do validation
    nrow = 6; ncol = 8
    count = 0

    xgrid = range(.01, stop=1.3, length=100)
    ygrid = Array{Float64, 1}(undef, 100)
    PyPlot.close("all") #close existing windows
    plt, axs = PyPlot.subplots(nrow, ncol)
    before = Dates.now()
    for j = 1:nrow*ncol
        mod(j, 10) == 0 ? (@info j) : nothing
        if parsed_args["fast"]
            (pdf_raw, cdf_raw, dpdf_raw, quantInfo_raw) = solve(btg1, validate = j)
            pdf1, cdf1, dpdf1, quantbound1, support1 = pre_process(x[1, :], Fx[1, :], pdf_raw, cdf_raw, dpdf_raw, quantInfo_raw)   
        else
            (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, j)
            btg2 = btg(trainingdata_minus_i, rangeθm, rangeλ; quadtype = ["SparseGrid", "Gaussian"])
            (pdf_raw, cdf_raw, dpdf_raw, quantInfo_raw) = solve(btg2)
            pdf1, cdf1, dpdf1, quantbound1, support1 = pre_process(x_i, Fx_i, pdf_raw, cdf_raw, dpdf_raw, quantInfo_raw)
        end
        median1 = quantile(cdf1, quantbound1, support1)[1]
        CI1 = credible_interval(cdf1, quantbound1, support1; mode=:equal, wp=.95)[1]
        yj_true = getLabel(btg1.trainingData)[j]
        count += (yj_true >= CI1[1])&&(yj_true <= CI1[2]) ? 1 : 0
        ygrid = pdf1.(xgrid)
        # plot
        ind1 = Int64(ceil(j/ncol))
        ind2 = Int64(j - ncol*(floor((j-.1)/ncol)))
        axs[ind1, ind2].plot(xgrid, ygrid, linewidth = 1.0, linestyle = "-")
        # axs[ind1, ind2].plot(xgrid, cdf1.(xgrid), color = "orange", linewidth = 1.0, linestyle = "-")
        axs[ind1, ind2].vlines(yj_true, 0, pdf1(yj_true), label = "true value")
        axs[ind1, ind2].vlines(median1, 0, pdf1(median1), label = "median")
        CI_id = (xgrid .> CI1[1]) .* (xgrid .< CI1[2])
        CI_xrange = vcat(CI1[1], xgrid[CI_id], CI1[2]) 
        CI_yrange = vcat(pdf1(CI1[1]), ygrid[CI_id], pdf1(CI1[2])) # utilize previous evaluation results
        axs[ind1, ind2].fill_between(CI_xrange, 0, CI_yrange, alpha = 0.3, label = "95% confidence interval")
    end 
    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
    count /= nrow*ncol # percentage that y_true falls into computed CI
    PyPlot.suptitle("Cross Validation $(parsed_args["fast"]), CI accuracy  $(@sprintf("%.4f", count)) ", fontsize=10)

    io2 = open("Exp_abalone_validate.txt", "a") 
    write(io2, "$(Dates.now())  ;    $myquadtype     ;   $rangeθ  ;  $rangeλ  ;  $(parsed_args["fast"])  ;   $elapsedmin ; $count \n")
    close(io2)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs
        ax.label_outer()
    end
    PyPlot.savefig("figure/test_v6_ind_$(myquadtype[1])$(myquadtype[2])_rθ_$(Int(rangeθ[1]))_$(Int(rangeθ[2]))_rλ_$(Int(rangeλ[1]))_$(Int(rangeλ[2]))_$(parsed_args["fast"]).pdf")
end

