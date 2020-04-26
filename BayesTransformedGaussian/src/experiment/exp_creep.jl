using Dates
using ArgParse
using Printf
using Random
using GaussianProcesses

include("../btg.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "--test"
        help = "test or not"
        action = :store_true
    "--validate"
        help = "do cross validation or not"
        action = :store_true
    "--fast"
        help = "use fast or not"
        action = :store_true
    "--GP"
        help = "test GP model"
        action = :store_true
    "--logGP"
        help = "test log-GP model"
        action = :store_true
end
parsed_args = parse_args(ARGS, s)

# load creep data
df = DataFrame(CSV.File("../datasets/creeprupt/taka", header=0))
data = convert(Matrix, df[:,[1; 3:end]])
target = convert(Array, df[:, 2]) 
# shuffle data
ind_shuffle = randperm(MersenneTwister(123), size(data, 1)) 
data = data[ind_shuffle, :]
target = target[ind_shuffle]
# training set
id_train = 1:400; posx = 1:30; posc = 1:30; n_train = length(id_train)
x = data[id_train, posx] 
Fx = data[id_train, posc] 
y = float(target[id_train])
ymax_train = maximum(y)
y ./= ymax_train
trainingData0 = trainingData(x, Fx, y) #training data used for testing various functions
d = getDimension(trainingData0); n = getNumPts(trainingData0); p = getCovDimension(trainingData0)

#parameter setting
rangeθ = [0.125 1000]
rangeθm = repeat(rangeθ, d, 1)
rangeλ = [-1.5 1.] #we will always used 1 range scale for lambda
myquadtype = ["QuasiMonteCarlo", "QuasiMonteCarlo"]

# build btg model
btg0 = btg(trainingData0, rangeθm, rangeλ; quadtype = myquadtype)
(pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw) = solve(btg0); #initialize training_buffer_dicts, solve once so can use fast techiques to extrapolate submatrix determinants, etc.

####################################
############### Test ###############
####################################
if parsed_args["test"]
    @info "Start Test"
    id_test = 801:length(target)
    n_test = length(id_test)
    id_fail = []
    id_nonproper = []
    x_test = data[id_test, posx]
    Fx_test = data[id_test, posc]
    y_test_true = target[id_test]
    count_test = 0
    error_abs = 0.
    error_sq = 0.
    nlpd = 0.
    for i in 1:n_test
        global error_abs, error_sq, nlpd, count_test
        mod(i, 10) == 0 ? (@info i) : nothing
        # @info "i" i
        x_test_i = reshape(x_test[i, :], 1, length(posx))
        Fx_test_i = reshape(Fx_test[i, :], 1, length(posc))
        try
            pdf_test_i, cdf_test_i, dpdf_test_i, quantbound_test_i, support_test_i = pre_process(x_test_i, Fx_test_i, pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw)
            y_test_i_true = y_test_true[i]
            median_test_i = ymax_train * quantile(cdf_test_i, quantbound_test_i, support_test_i)[1]
            try 
                CI_test_i = ymax_train .* credible_interval(cdf_test_i, quantbound_test_i, support_test_i; mode=:equal, wp=.95)[1]
                count_test += (y_test_i_true >= CI_test_i[1])&&(y_test_i_true <= CI_test_i[2]) ? 1 : 0
            catch err
                append!(id_fail, i)
            end
            error_abs += abs(y_test_i_true - median_test_i)
            error_sq += (y_test_i_true - median_test_i)^2
            nlpd += log(pdf_test_i(y_test_i_true)) 
            # @info y_test_i_true, median_test_i
            # @info error_abs, error_sq
            # @info count_test, id_fail, id_nonproper
        catch err 
            append!(id_nonproper, i)
        end
    end
    count_test /= n_test - length(id_fail)
    error_abs  /= n_test
    error_sq   /= n_test
    nlpd       /= -n_test

    if parsed_args["GP"] 
        @info "Start GP"
        global error_abs_GP, error_sq_GP, CI_test_GP, count_test_GP, nlpd_GP
        # training set
        x = data[id_train, posx]' 
        y = float(target[id_train])
        # build and fit a GP
        # mymean = MeanLin(zeros(d));
        mymean = MeanZero() 
        kern = SE(zeros(d),0.0) 
        gp = GP(x, y, mymean, kern) 
        optimize!(gp)     
        # predict
        x_test = data[id_test, posx]'
        y_test_true = target[id_test]
        μ, σ² = predict_y(gp, x_test); stdv = sqrt.(σ²)
        error_GP = abs.(μ .- y_test_true)
        error_abs_GP = mean(error_GP)
        error_sq_GP = mean(error_GP.^2)
        CI_test_GP = hcat(-1.96.*stdv .+ μ, 1.96.*stdv .+ μ)
        count_test_GP = sum((y_test_true .>= CI_test_GP[:, 1]) .* (y_test_true .<= CI_test_GP[:,2]))/n_test
        nlpd_GP = -mean(log.(pdf.(Normal(), (y_test_true.-μ)./stdv)./stdv))
    end

    if parsed_args["logGP"]
        @info "Start logGP"
        global error_abs_logGP, error_sq_logGP, CI_test_logGP, count_test_logGP, nlpd_logGP
        x = data[id_train, posx]' 
        y = float(target[id_train])
        trans = BoxCox()
        g_fixed(x) = trans(x, 0.); dg(x) = partialx(trans, x, 0.)
        invg(x) = inverse(trans, x, 0.)
        gy = g_fixed.(y) 
        # build and fit a GP
        # mymean = MeanLin(zeros(d))
        mymean = MeanZero() 
        kern = SE(zeros(d),0.0) 
        loggp = GP(x, gy, mymean, kern) 
        optimize!(loggp) 
        # predict
        x_test = data[id_test, posx]'
        y_test_true = target[id_test]
        μ, σ² = predict_y(loggp, x_test); stdv = sqrt.(σ²)
        CI_test_logGP = invg.(hcat(-1.96.*stdv .+ μ, 1.96.*stdv .+ μ))
        count_test_logGP = sum((y_test_true .>= CI_test_logGP[:, 1]) .* (y_test_true .<= CI_test_logGP[:,2]))/n_test
        y_pred = invg.(μ)
        error_logGP = abs.(y_pred .- y_test_true)
        error_abs_logGP = mean(error_logGP)
        error_sq_logGP = mean(error_logGP.^2)
        nlpd_logGP = -mean(log.( dg.(y_test_true) .* pdf.(Normal(), (g_fixed.(y_test_true).-μ)./stdv) ./stdv ))
    end
    
    io1 = open("Exp_creep_test.txt", "a") 
    write(io1, "\n$(Dates.now()) \n" )
    write(io1, "Data set: creep   
        id_train:  $id_train;  id_test:  $id_test  \n") 
    write(io1, "BTG model:  
        $myquadtype  ;  rangeθ: $rangeθ;   rangeλ: $rangeλ \n")
    if parsed_args["GP"] && parsed_args["logGP"]
        write(io1, "Compare test results: ")
        write(io1, "                               BTG               GP               logGP
        credible intervel accuracy percentage:   $(@sprintf("%11.8f", count_test))       $(@sprintf("%11.8f", count_test_GP))       $(@sprintf("%11.8f", count_test_logGP)) 
        mean absolute error:                     $(@sprintf("%11.8f", error_abs))       $(@sprintf("%11.8f", error_abs_GP))       $(@sprintf("%11.8f", error_abs_logGP))  
        mean squared error:                      $(@sprintf("%11.8f", error_sq))       $(@sprintf("%11.8f", error_sq_GP))       $(@sprintf("%11.8f", error_sq_logGP))   
        mean negative log predictive density:    $(@sprintf("%11.8f", nlpd))       $(@sprintf("%11.8f", nlpd_GP))       $(@sprintf("%11.8f", nlpd_logGP))  
        BTG: Failed index in credible intervel:  $id_fail 
        BTG: Failed index in pdf computation:    $id_nonproper\n")
    else
        write(io1, "BTG test results: 
        credible intervel accuracy percentage:   $(@sprintf("%11.8f", count_test))     
        mean absolute error:                     $(@sprintf("%11.8f", error_abs))   
        mean squared error:                      $(@sprintf("%11.8f", error_sq)) 
        mean negative log predictive density:    $(@sprintf("%11.8f", nlpd))   
        BTG: Failed index in credible intervel:  $id_fail 
        BTG: Failed index in pdf computation:    $id_nonproper\n")
    end
    close(io1)


end

####################################
############ Validation ############
####################################
if parsed_args["validate"]
    @info "Start validation"
    if filesize("Exp_creep_validate.txt") == 0
        # write the setting headers
        io2 = open("Exp_creep_validate.txt", "w") 
        write(io2, "            Time            ;    ind_train    ;            quadtype            ;     rangeθ    ;    rangeλ   ;  fast ;  elapsedmin ; CI accuracy\n")
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
    count_val = 0

    xgrid = range(.01, stop=1.3, length=100)
    ygrid = Array{Float64, 1}(undef, 100)
    PyPlot.close("all") #close existing windows
    plt, axs = PyPlot.subplots(nrow, ncol)
    before = Dates.now()
    for i = 1:nrow*ncol
        global count_val
        mod(j, 10) == 0 ? (@info j) : nothing
        if parsed_args["fast"]
            (pdf_val_i_raw, cdf_val_i_raw, dpdf_val_i_raw, quantInfo_val_i_raw) = solve(btg0, validate = i)
            pdf_val_i, cdf_val_i, dpdf_val_i, quantbound_val_i, support_val_i = pre_process(x[1, :], Fx[1, :], pdf_val_i_raw, cdf_val_i_raw, dpdf_val_i_raw, quantInfo_val_i_raw)   
        else
            (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, i)
            btg_val_naive = btg(trainingdata_minus_i, rangeθm, rangeλ; quadtype = myquadtype)
            (pdf_val_i_raw, cdf_val_i_raw, dpdf_val_i_raw, quantInfo_val_i_raw) = solve(btg_val_naive)
            pdf_val_i, cdf_val_i, dpdf_val_i, quantbound_val_i, support_val_i  = pre_process(x_i, Fx_i, pdf_val_i_raw, cdf_val_i_raw, dpdf_val_i_raw, quantInfo_val_i_raw)
        end
        median_val_i = quantile(cdf_val_i, quantbound_val_i, support_val_i)[1]
        CI_val_i = credible_interval(cdf_val_i, quantbound_val_i, support_val_i; mode=:equal, wp=.95)[1]
        y_val_i_true = getLabel(btg0.trainingData)[i]
        count_val += (y_val_i_true >= CI_val_i[1])&&(y_val_i_true <= CI_val_i[2]) ? 1 : 0
        ygrid = pdf_val_i.(xgrid)
        # plot
        ind1 = Int64(ceil(j/ncol))
        ind2 = Int64(j - ncol*(floor((j-.1)/ncol)))
        axs[ind1, ind2].plot(xgrid, ygrid, linewidth = 1.0, linestyle = "-")
        # axs[ind1, ind2].plot(xgrid, cdf1.(xgrid), color = "orange", linewidth = 1.0, linestyle = "-")
        axs[ind1, ind2].vlines(y_val_i_true, 0, pdf_val_i(y_val_i_true), label = "true value")
        axs[ind1, ind2].vlines(median_val_i, 0, pdf_val_i(median_val_i), label = "median")
        CI_id = (xgrid .> CI_val_i[1]) .* (xgrid .< CI_val_i[2])
        CI_xrange = vcat(CI_val_i[1], xgrid[CI_id], CI_val_i[2]) 
        CI_yrange = vcat(pdf_val_i(CI_val_i[1]), ygrid[CI_id], pdf_val_i(CI_val_i[2])) # utilize previous evaluation results
        axs[ind1, ind2].fill_between(CI_xrange, 0, CI_yrange, alpha = 0.3, label = "95% confidence interval")
    end 
    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
    count_val /= nrow*ncol # percentage that y_true falls into computed CI
    PyPlot.suptitle("Cross Validation $(parsed_args["fast"]), CI accuracy  $(@sprintf("%.4f", count)) ", fontsize=10)

    io2 = open("Exp_creep_validate.txt", "a") 
    write(io2, "$(Dates.now())  ;  $id_train   ; $myquadtype ;   $rangeθ  ;  $rangeλ  ;  $(parsed_args["fast"])  ;   $elapsedmin ; $count_val \n")
    close(io2)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs
        ax.label_outer()
    end
    PyPlot.savefig("figure/exp_creep_$(ind[1])_$(ind[2])_$(myquadtype[1])$(myquadtype[2])_rθ_$(Int(rangeθ[1]))_$(Int(rangeθ[2]))_rλ_$(Int(rangeλ[1]))_$(Int(rangeλ[2]))_$(parsed_args["fast"]).pdf")
end

