using Dates
using ArgParse
using Printf
using Random
using GaussianProcesses

# before_all = Dates.now()

include("../btg.jl")
s = ArgParseSettings()
# The defaut setting: --test: multiple length scale, QMC
@add_arg_table! s begin
    "--test"
        help = "test or not"
        action = :store_true
    "--single"
        help = "use single length scale or not"
        action = :store_true 
    "--sparse"   
        help = "use SparseGrid or not"
        action = :store_true
    "--validate"
        help = "do cross validation or not"
        action = :store_true
    "--fast"
        help = "use fast validation or not"
        action = :store_true
    "--GP"
        help = "test GP model"
        action = :store_true
    "--logGP"
        help = "test log-GP model"
        action = :store_true
    "--posc"
        help = "another option with an argument"
        arg_type = Int
        default = 7
    "--singletest"
        help = "write log to single test"
        action = :store_true
    "--ntrain"
        help = "another option with an argument"
        arg_type = Int
        default = 200
    "--ntest"
        help = "another option with an argument"
        arg_type = Int
        default = 5
    "--quadtype"
        help = "quadrature type for theta"
        arg_type = Int
        default = 1
end
parsed_args = parse_args(ARGS, s)
# load abalone data
df = DataFrame(CSV.File("../datasets/abalone.csv"))
data = convert(Matrix, df[:,2:8]) #length, diameter, height, whole weight, shucked weight, viscera weight, shell weight
target = convert(Array, df[:, 9]) #age
# shuffle data
randseed = 1234; rng = MersenneTwister(randseed)
ind_shuffle = randperm(rng, size(data, 1)) 
data = data[ind_shuffle, :]
target = target[ind_shuffle]
# training set
id_train = 1:parsed_args["ntrain"]; posx = 1:7; posc = 1:parsed_args["posc"]
n_train = length(id_train)
x = data[id_train, posx] 
Fx = data[id_train, posc] 
y = float(target[id_train])
ymax_train = maximum(y)
y ./= ymax_train
trainingData0 = trainingData(x, Fx, y) 
d = getDimension(trainingData0); n = getNumPts(trainingData0); p = getCovDimension(trainingData0)

#parameter setting
if parsed_args["quadtype"] == 1
    myquadtype = ["Gaussian", "Gaussian"]
elseif parsed_args["quadtype"] == 2
    myquadtype = ["SparseGrid", "Gaussian"]
elseif parsed_args["quadtype"] == 3
    myquadtype = ["SparseCarlo", "SparseCarlo"]
else
    myquadtype = ["QuasiMonteCarlo", "QuasiMonteCarlo"]
end
rangeλ = [-1.5 1.] 
rangeθs = [500. 1000]
rangeθm = repeat(rangeθs, d, 1)
rangeθ = parsed_args["single"] ? rangeθs : rangeθm
# build btg model
btg0 = btg(trainingData0, rangeθ, rangeλ; quadtype = myquadtype)
(pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw) = solve(btg0);

####################################
############### Test ###############
####################################
if parsed_args["test"]
    @info "Start Test"
    before = Dates.now()
    id_test = 1001:(parsed_args["ntest"]+1000)
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
    error_abs_set = zeros(n_test)
    nlpd_set = zeros(n_test)
    for i in 1:n_test
        global error_abs, error_sq, nlpd, count_test
        mod(i, 20) == 0 ? (@info i) : nothing
        # @info "i" i
        x_test_i = reshape(x_test[i, :], 1, length(posx))
        Fx_test_i = reshape(Fx_test[i, :], 1, length(posc))
        try
            pdf_test_i, cdf_test_i, dpdf_test_i, quantbound_test_i, support_test_i = pre_process(x_test_i, Fx_test_i, pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw)
            y_test_i_true = y_test_true[i]
            median_test_i = ymax_train * quantile(cdf_test_i, quantbound_test_i, support_test_i)[1]
            # @info "True, median " y_test_i_true, median_test_i
            try 
                CI_test_i = ymax_train .* credible_interval(cdf_test_i, quantbound_test_i, support_test_i; mode=:equal, wp=.95)[1]
                count_test += (y_test_i_true >= CI_test_i[1])&&(y_test_i_true <= CI_test_i[2]) ? 1 : 0
                # @info "CI" CI_test_i
            catch err
                append!(id_fail, i)
            end
            error_abs += abs(y_test_i_true - median_test_i)
            error_sq += (y_test_i_true - median_test_i)^2
            nlpd += log(pdf_test_i(y_test_i_true)) 
            error_abs_set[i] = error_abs
            nlpd_set[i] = nlpd
        # @info "Count, id_fail" count_test, id_fail
        catch err 
            append!(id_nonproper, i)
        end
    end
    count_test /= n_test - length(id_fail) - length(id_nonproper)
    error_abs  /= n_test - length(id_nonproper)
    error_sq   /= n_test - length(id_nonproper)
    nlpd       /= -n_test - length(id_nonproper)
    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)

    io = open("Exp_abalone_errorhist.txt", "a") 
    write(io, "\n$(Dates.now()), randseed: $randseed \n")
    write(io, "Data set: Abalone   
    id_train:  $id_train;  id_test:  $id_test;   posx: $posx;   posc: $posc\n") 
    write(io, "BTG model:  
            $myquadtype  ;  rangeλ: $rangeλ;   rangeθ: $rangeθs (single length-scale: $(parsed_args["single"])) \n")
    write(io, "Absolute error history \n $error_abs_set \n")
    write(io, "Negative log predictive density history \n $nlpd_set \n")
    close(io)

    if parsed_args["GP"] 
        global error_abs_GP, error_sq_GP, CI_test_GP, count_test_GP, nlpd_GP
        # training set
        x = data[id_train, posx]' 
        y = float(target[id_train])
        # build and fit a GP
        mymean = MeanLin(zeros(d))
        # mymean = MeanZero() 
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
        global error_abs_logGP, error_sq_logGP, CI_test_logGP, count_test_logGP, nlpd_logGP
        x = data[id_train, posx]' 
        y = float(target[id_train])
        trans = BoxCox()
        g_fixed(x) = trans(x, 0.); dg(x) = partialx(trans, x, 0.)
        invg(x) = inverse(trans, x, 0.)
        gy = g_fixed.(y) 
        # build and fit a GP
        mymean = MeanLin(zeros(d))
        # mymean = MeanZero() 
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

    if parsed_args["singletest"]
        @info "Average time for single test: " elapsedmin/n_test
        io1 = open("Exp_abalone_singletest.txt", "a")
        write(io1, "\n$(Dates.now()), randseed: $randseed \n" )
        write(io1, "Data set: Abalone   
            id_train:  $id_train;  id_test:  $id_test;   posx: $posx;   posc: $posc\n") 
        write(io1, "BTG model:  
            $myquadtype  ;  rangeλ: $rangeλ;   rangeθ: $rangeθs (single length-scale: $(parsed_args["single"])) \n")
        write(io1, "BTG test results: 
            credible intervel accuracy percentage:   $(@sprintf("%11.8f", count_test))     
            mean absolute error:                     $(@sprintf("%11.8f", error_abs))   
            mean squared error:                      $(@sprintf("%11.8f", error_sq)) 
            mean negative log predictive density:    $(@sprintf("%11.8f", nlpd))
            Time cost by prediction: $elapsedmin   
            Failed index in credible intervel:       $id_fail 
            BTG: Failed index in pdf computation:     $id_nonproper\n")
        close(io1)
    else # write results 
        io1 = open("Exp_abalone_test.txt", "a") 
        write(io1, "\n$(Dates.now()), randseed: $randseed \n" )
        write(io1, "Data set: Abalone   
            id_train:  $id_train;  id_test:  $id_test;   posx: $posx;   posc: $posc\n") 
        write(io1, "BTG model:  
            $myquadtype  ;  rangeλ: $rangeλ;   rangeθ: $rangeθs (single length-scale: $(parsed_args["single"])) \n")
        if parsed_args["GP"] && parsed_args["logGP"]
            write(io1, "Compare test results: ")
            write(io1, "                               BTG               GP               logGP
            credible intervel accuracy percentage:   $(@sprintf("%11.8f", count_test))       $(@sprintf("%11.8f", count_test_GP))       $(@sprintf("%11.8f", count_test_logGP)) 
            mean absolute error:                     $(@sprintf("%11.8f", error_abs))       $(@sprintf("%11.8f", error_abs_GP))       $(@sprintf("%11.8f", error_abs_logGP))  
            mean squared error:                      $(@sprintf("%11.8f", error_sq))       $(@sprintf("%11.8f", error_sq_GP))       $(@sprintf("%11.8f", error_sq_logGP))   
            mean negative log predictive density:    $(@sprintf("%11.8f", nlpd))       $(@sprintf("%11.8f", nlpd_GP))       $(@sprintf("%11.8f", nlpd_logGP))  
            Time cost by prediction: $elapsedmin
            Time cost by single prediction: $(elapsedmin/n_test)
            BTG: Failed index in credible intervel:   $id_fail 
            BTG: Failed index in pdf computation:     $id_nonproper\n")
        else
            write(io1, "BTG test results: 
            credible intervel accuracy percentage:   $(@sprintf("%11.8f", count_test))     
            mean absolute error:                     $(@sprintf("%11.8f", error_abs))   
            mean squared error:                      $(@sprintf("%11.8f", error_sq)) 
            mean negative log predictive density:    $(@sprintf("%11.8f", nlpd))
            Time cost by prediction: $elapsedmin   
            Time cost by single prediction: $(elapsedmin/n_test)
            Failed index in credible intervel:       $id_fail 
            BTG: Failed index in pdf computation:     $id_nonproper\n")
        end
        close(io1)
    end


end

####################################
############ Validation ############
####################################
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

mutable struct validation_plot_buffer
    ygrid::Array{Float64, 1}
    median::Float64
    CI::Array{Float64, 1}
    pdf_ytrue::Float64
    pdf_median_i::Float64
    pdf_CI::Array{Float64, 1}
    function validation_plot_buffer(ygrid_i, median_i, CI_i, pdf_ytrue_i, pdf_median_i, pdf_CI_i)
        return new(ygrid_i, median_i, CI_i, pdf_ytrue_i, pdf_median_i, pdf_CI_i)
    end
end

function unpack(a::validation_plot_buffer) 
    return (a.ygrid, a.median, a.CI, a.pdf_ytrue, a.pdf_median_i, a.pdf_CI)
end

if parsed_args["validate"]
    @info "Start validation"
    before = Dates.now()
    id_fail_val = []
    id_nonproper_val = []
    count_val = 0
    error_abs_val = 0.
    error_sq_val = 0.
    nlpd_val = 0.
    # prepare space for plotting
    n_row = 6; n_col = 8
    xgrid = range(.01, stop=1.3, length=100)
    validation_plot_buffer_dict = Dict{T where T<:Int64, validation_plot_buffer}()
    for i = 1:n_train
        global count_val, error_abs_val, error_sq_val, nlpd_val
        mod(i, 10) == 0 ? (@info i) : nothing
        if parsed_args["fast"]
            (pdf_val_i_raw, cdf_val_i_raw, dpdf_val_i_raw, quantInfo_val_i_raw) = solve(btg0, validate = i)
        else
            (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData0, i)
            btg_val_naive = btg(trainingdata_minus_i, rangeθ, rangeλ; quadtype = myquadtype)
            (pdf_val_i_raw, cdf_val_i_raw, dpdf_val_i_raw, quantInfo_val_i_raw) = solve(btg_val_naive)
        end
        try
            x_i = x[i:i, :]; Fx_i = Fx[i:i, :]
            pdf_val_i, cdf_val_i, dpdf_val_i, quantbound_val_i, support_val_i = pre_process(x_i, Fx_i, pdf_val_i_raw, cdf_val_i_raw, dpdf_val_i_raw, quantInfo_val_i_raw)   
            median_val_i = quantile(cdf_val_i, quantbound_val_i, support_val_i)[1]
            y_val_i_true = getLabel(btg0.trainingData)[i]
            pdf_ytrue_i = pdf_val_i(y_val_i_true)
            try
                CI_val_i = credible_interval(cdf_val_i, quantbound_val_i, support_val_i; mode=:equal, wp=.95)[1]
                count_val += (y_val_i_true >= CI_val_i[1])&&(y_val_i_true <= CI_val_i[2]) ? 1 : 0
                if length(validation_plot_buffer_dict) < n_row*n_col
                    ygrid_i = pdf_val_i.(xgrid)
                    pdf_CI_i = pdf_val_i.(CI_val_i)
                    pdf_median_i = pdf_val_i.(median_val_i)
                    push!(validation_plot_buffer_dict, i => validation_plot_buffer(ygrid_i, median_i, CI_val_i, pdf_ytrue_i, pdf_median_i, pdf_CI_i))
                end
            catch err
                append!(id_fail_val, i)
            end
            error_abs_val += abs(y_val_i_true - median_val_i)
            error_sq_val += (y_val_i_true - median_val_i)^2
            nlpd_val -= log(pdf_ytrue_i) 
        # @info "Count, id_fail" count_test, id_fail
        catch err 
            append!(id_nonproper_val, i)
        end    
    end 
    count_val      /= n_train - length(id_fail_val) - length(id_nonproper_val)
    error_abs_val  /= n_train - length(id_nonproper_val)
    error_sq_val   /= n_train - length(id_nonproper_val)
    nlpd_val       /= n_train - length(id_nonproper_val)
    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)

    # save results
    io2 = open("Exp_abalone_LOOCV.txt", "a") 
    write(io2, "\n$(Dates.now()), randseed: $randseed \n" )
    write(io2, "Data set: Abalone   
        id_train:  $id_train;   posx: $posx;   posc: $posc\n") 
    write(io2, "BTG model:  
        $myquadtype  ;  rangeλ: $rangeλ;   rangeθ: $rangeθs (single length-scale: $(parsed_args["single"])) 
        Fast LOOCV: $(parsed_args["fast"]) \n")
    write(io2, "LOOCV on training set results: 
    credible intervel accuracy percentage:   $(@sprintf("%11.8f", count_val))     
    mean absolute error:                     $(@sprintf("%11.8f", error_abs_val))   
    mean squared error:                      $(@sprintf("%11.8f", error_sq_val)) 
    mean negative log predictive density:    $(@sprintf("%11.8f", nlpd_val))
    Time validation took: $elapsedmin   
    Failed index in credible intervel:       $id_fail_val 
    BTG: Failed index in pdf computation:     $id_nonproper_val\n")
    close(io2)

    # Plot the first 48
    PyPlot.close("all") #close existing windows
    plt, axs = PyPlot.subplots(n_row, n_col)
    PyPlot.suptitle("Partial Plots of LOOCV", fontsize=10)
    id = 0
    for i in keys(validation_plot_buffer_dict)
        id += 1
        global id
        ind1 = Int64(ceil(id/n_col))
        ind2 = Int64(id - n_col*(floor((id-.1)/n_col)))
        y_val_i_true = getLabel(btg0.trainingData)[i]
        ygrid_i, median_i, CI_val_i, pdf_ytrue_i, pdf_median_i, pdf_CI_i = unpack(validation_plot_buffer_dict[i])
        axs[ind1, ind2].plot(xgrid, ygrid_i, linewidth = 1.0, linestyle = "-")
        # axs[ind1, ind2].plot(xgrid, cdf1.(xgrid), color = "orange", linewidth = 1.0, linestyle = "-")
        axs[ind1, ind2].vlines(y_val_i_true, 0, pdf_ytrue_i, label = "true value")
        axs[ind1, ind2].vlines(median_val_i, 0, pdf_median_i, label = "median")
        CI_id = (xgrid .> CI_val_i[1]) .* (xgrid .< CI_val_i[2])
        CI_xrange = vcat(CI_val_i[1], xgrid[CI_id], CI_val_i[2]) 
        CI_yrange = vcat(pdf_CI_i[1], ygrid_i[CI_id], pdf_CI_i[2]) # utilize previous evaluation results
        axs[ind1, ind2].fill_between(CI_xrange, 0, CI_yrange, alpha = 0.3, label = "95% confidence interval")
    end
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs
        ax.label_outer()
    end
    PyPlot.savefig("figure/exp_abalone_$(id_train[1])_$(id_train[2])_$(myquadtype[1])$(myquadtype[2])_posc$(posc[2])_rθ_$(Int(rangeθs[1]))_$(Int(rangeθs[2]))_rλ_$(Int(rangeλ[1]))_$(Int(rangeλ[2]))_$(parsed_args["fast"]).pdf")

end

# after_all = Dates.now()
# elapsedmin_all = round(((after_all - before_all) / Millisecond(1000))/60, digits=5)
# @info "Total time" elapsedmin_all