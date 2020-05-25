using Dates
using ArgParse
using Printf
using Random
using GaussianProcesses

# before_all = Dates.now()

include("../../btg.jl")
s = ArgParseSettings()
# The defaut setting: --test: multiple length scale, QMC
@add_arg_table! s begin
    "--test"
        help = "test or not"
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
    "--p"
        help = "degree of covariate function"
        arg_type = Int
        default = 1
    "--singletest"
        help = "write log to single test"
        action = :store_true
    "--ntrain"
        help = "number of training points"
        arg_type = Int
        default = 101
    "--ntest"
        help = "number of testing points"
        arg_type = Int
        default = 401
    "--noiselevel"
        help = "noise level in observation"
        arg_type = Int
        default = 1
    "--lmin"
        help = "minimal length scale"
        arg_type = Float64
        default = 0.5
    "--lmax"
        help = "maximum length scale"
        arg_type = Float64
        default = 2.  
    "--randseed"
        help = "noise level in observation"
        arg_type = Int
        default = 1234
end
parsed_args = parse_args(ARGS, s)

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
target = (sin.(data) .+ 0.1 * parsed_args["noiselevel"].* randn(rng, n_train) .+ 10).^(1/3)
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
y_test_true = (sin.(x_test) .+ 10).^(1/3)

#parameter setting
# myquadtype = parsed_args["sparse"] ? ["SparseCarlo", "SparseCarlo"] : ["QuasiMonteCarlo", "QuasiMonteCarlo"]
myquadtype = ["Gaussian", "Gaussian"]
rangeλ = [-1.5 1.] 
lmin = parsed_args["lmin"]
lmax = parsed_args["lmax"]
rangeθ = [1/lmax^2 1/lmin^2]
@info "rangeθ, rangel:" rangeθ, [lmin lmax]
# rangeθ = [0.111 25]
# build btg model
btg0 = btg(trainingData0, rangeθ, rangeλ; quadtype = myquadtype)
(pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw) = solve(btg0);

####################################
############### Test ###############
####################################
if parsed_args["test"]
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
    end
    count_test /= n_test - length(id_fail) - length(id_nonproper)
    error_abs  /= n_test - length(id_nonproper)
    error_sq   /= n_test - length(id_nonproper)
    nlpd       /= -n_test - length(id_nonproper)
    after = Dates.now()
    elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)

    # Plot
    PyPlot.close("all") #close existing windows
    PyPlot.plot(x_test, median_set, label = "BTG median")
    PyPlot.plot(x_test, y_test_true, label = "true")
    PyPlot.fill_between(dropdims(x_test; dims = 2), CI_set[:, 1], CI_set[:, 2], alpha = 0.3, label = "95% confidence interval")
    PyPlot.scatter(x, target, s = 10, c = "k", marker = "*")
    PyPlot.legend(fontsize=8)
    PyPlot.grid()
    PyPlot.title("BTG $myquadtype", fontsize=10)
    PyPlot.savefig("exp_synthetic1_btg_range_$(Int(10*lmin))_$(Int(10*lmax))_noise$(parsed_args["noiselevel"])_randseed_$(randseed)_p$(parsed_args["p"]).pdf")


    if parsed_args["GP"] 
        @info "Start GP"
        global error_abs_GP, error_sq_GP, CI_test_GP, count_test_GP, nlpd_GP
        # training set
        x = reshape(x, 1, n_train) 
        y = target
        # build and fit a GP
        # mymean = MeanLin(zeros(d))
        mymean = MeanZero() 
        kern = SE(zeros(d),0.0) 
        gp = GP(x, y, mymean, kern) 
        optimize!(gp)     
        # predict
        x_test = reshape(x_test, 1, n_test)
        μ, σ² = predict_y(gp, x_test); stdv = sqrt.(σ²)
        error_GP = abs.(μ .- y_test_true)
        error_abs_GP = mean(error_GP)
        error_sq_GP = mean(error_GP.^2)
        CI_test_GP = hcat( -1.96 .* stdv .+ μ, 1.96 .* stdv .+ μ)
        count_test_GP = sum((y_test_true .>= CI_test_GP[:, 1]) .* (y_test_true .<= CI_test_GP[:,2]))/n_test
        nlpd_GP = -mean(log.(pdf.(Normal(), (y_test_true.-μ)./stdv)./stdv))
        # Plot
        # PyPlot.close("all") #close existing windows
        # PyPlot.plot(dropdims(x_test; dims = 1), μ, label = "GP median")
        # PyPlot.scatter(x, target, s = 10, c = "k", marker = "*")
        # PyPlot.fill_between(dropdims(x_test; dims = 1), CI_test_GP[:, 1], CI_test_GP[:, 2], alpha = 0.3, label = "95% credibel interval")
        # PyPlot.legend(fontsize=8)
        # PyPlot.grid()
        # PyPlot.title("GP", fontsize=10)
        # PyPlot.savefig("exp_synthetic1_GP_range_$(Int(10*lmin))_$(Int(10*lmax))_noise$(parsed_args["noiselevel"]).pdf")
    end

    if parsed_args["logGP"]
        @info "Start logGP"
        global error_abs_logGP, error_sq_logGP, CI_test_logGP, count_test_logGP, nlpd_logGP
        x = reshape(x, 1, n_train) 
        y = target
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
        x_test = reshape(x_test, 1, n_test)
        μ, σ² = predict_y(loggp, x_test); stdv = sqrt.(σ²)
        CI_test_logGP = invg.(hcat(-1.96.*stdv .+ μ, 1.96.*stdv .+ μ))
        count_test_logGP = sum((y_test_true .>= CI_test_logGP[:, 1]) .* (y_test_true .<= CI_test_logGP[:,2]))/n_test
        y_pred = invg.(μ)
        error_logGP = abs.(y_pred .- y_test_true)
        error_abs_logGP = mean(error_logGP)
        error_sq_logGP = mean(error_logGP.^2)
        nlpd_logGP = -mean(log.( dg.(y_test_true) .* pdf.(Normal(), (g_fixed.(y_test_true).-μ)./stdv) ./stdv ))
        # Plot
        # PyPlot.close("all") #close existing windows
        # PyPlot.plot(dropdims(x_test; dims = 1), invg.(μ), label = "logGP median")
        # PyPlot.scatter(x, target, s = 10, c = "k", marker = "*")
        # PyPlot.fill_between(dropdims(x_test; dims = 1), CI_test_logGP[:, 1], CI_test_logGP[:, 2], alpha = 0.3, label = "95% credibel interval")
        # PyPlot.legend(fontsize=8)
        # PyPlot.grid()
        # PyPlot.title("logGP", fontsize=10)
        # PyPlot.savefig("exp_synthetic1_logGP_range_$(Int(10*lmin))_$(Int(10*lmax))_noise$(parsed_args["noiselevel"]).pdf")
    end

    # plot and Compare
    x_test = reshape(x_test, n_test)
    y_test_true = reshape(y_test_true, n_test)
    median_set = reshape(median_set, n_test)
    PyPlot.close("all") #close existing windows
    PyPlot.scatter(x, target, s = 10, c = "k", marker = "*")
    PyPlot.plot(x_test, y_test_true, label = "true")
    PyPlot.plot(x_test, median_set, label = "BTG median")
    # PyPlot.fill_between(x_test, CI_set[:, 1], CI_set[:, 2], alpha = 0.3, label = "95% CI BTG")
    # PyPlot.fill_between(x_test, CI_test_GP[:, 1], CI_test_GP[:, 2], alpha = 0.3, label = "95% CI GP")
    # PyPlot.fill_between(x_test, CI_test_logGP[:, 1], CI_test_logGP[:, 2], alpha = 0.3, label = "95% CI logGP")
    PyPlot.plot(x_test, CI_set[:, 1], "b-", label = "95% CI BTG")
    PyPlot.plot(x_test, CI_set[:, 2], "b-")
    PyPlot.plot(x_test, CI_test_GP[:, 1], "k:", label = "95% CI GP")
    PyPlot.plot(x_test, CI_test_GP[:, 2], "k:")
    PyPlot.plot(x_test, CI_test_logGP[:, 1], "c-.", label = "95% CI logGP")
    PyPlot.plot(x_test, CI_test_logGP[:, 2], "c-.")
    PyPlot.title("Compare BTG, GP and logGP", fontsize=10)
    PyPlot.legend(fontsize=8)
    PyPlot.grid()
    PyPlot.savefig("exp_synthetic1_compare_range_$(Int(10*lmin))_$(Int(10*lmax))_noise$(parsed_args["noiselevel"])_randseed_$(randseed)_p$(parsed_args["p"]).pdf")


    io1 = open("Exp_synthetic1_test.txt", "a") 
    write(io1, "\n$(Dates.now()) \n" )
    write(io1, "Data set: exp_synthetic1   
        train size:  $n_train;  test size:  $n_test;  dimension: $d;   degree of covariate functions: $(parsed_args["p"]) 
        randseed: $randseed;    noise level in training data: $(0.1*parsed_args["noiselevel"]) \n") 
    write(io1, "BTG model:  
        $myquadtype  ;  rangeλ: $rangeλ;  lengthscale: [$lmin, $lmax];  rangeθ: [$(@sprintf("%2.3f", rangeθ[1])), $(@sprintf("%2.3f", rangeθ[2]))] \n")
    if parsed_args["GP"] && parsed_args["logGP"]
        write(io1, "Compare test results: ")
        write(io1, "                               BTG               GP               logGP
        credible intervel accuracy percentage:   $(@sprintf("%11.8f", count_test))       $(@sprintf("%11.8f", count_test_GP))       $(@sprintf("%11.8f", count_test_logGP)) 
        mean absolute error:                     $(@sprintf("%11.8f", error_abs))       $(@sprintf("%11.8f", error_abs_GP))       $(@sprintf("%11.8f", error_abs_logGP))  
        mean squared error:                      $(@sprintf("%11.8f", error_sq))       $(@sprintf("%11.8f", error_sq_GP))       $(@sprintf("%11.8f", error_sq_logGP))   
        mean negative log predictive density:    $(@sprintf("%11.8f", nlpd))       $(@sprintf("%11.8f", nlpd_GP))       $(@sprintf("%11.8f", nlpd_logGP))  
        Time cost by prediction: $elapsedmin
        BTG: Failed index in credible intervel:   $id_fail 
        BTG: Failed index in pdf computation:     $id_nonproper\n")
    else
        write(io1, "BTG test results: 
        credible intervel accuracy percentage:   $(@sprintf("%11.8f", count_test))     
        mean absolute error:                     $(@sprintf("%11.8f", error_abs))   
        mean squared error:                      $(@sprintf("%11.8f", error_sq)) 
        mean negative log predictive density:    $(@sprintf("%11.8f", nlpd))
        Time cost by prediction: $elapsedmin   
        Failed index in credible intervel:       $id_fail 
        BTG: Failed index in pdf computation:     $id_nonproper\n")
    end
    close(io1)



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
    io2 = open("Exp_exp_synthetic1_LOOCV.txt", "a") 
    write(io2, "\n$(Dates.now()), randseed: $randseed \n" )
    write(io2, "Data set: exp_synthetic1   
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
    PyPlot.savefig("figure/exp_exp_synthetic1_$(id_train[1])_$(id_train[2])_$(myquadtype[1])$(myquadtype[2])_posc$(posc[2])_rθ_$(Int(rangeθs[1]))_$(Int(rangeθs[2]))_rλ_$(Int(rangeλ[1]))_$(Int(rangeλ[2]))_$(parsed_args["fast"]).pdf")

end

# after_all = Dates.now()
# elapsedmin_all = round(((after_all - before_all) / Millisecond(1000))/60, digits=5)
# @info "Total time" elapsedmin_all