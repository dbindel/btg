using Plots
include("bitcoin_btg.jl")
include("../../computation/tdist_mle.jl")

path = "../../datasets/"
data, target = bitcoin(path)

btg0 = load_bitcoin_btg()
(pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw) = solve(btg0);
f = theta -> tdist_mle(btg0, theta, 1.0);

####################################
######## Optional Plotting #########
####################################
if false
    Plots.plot()
    plt(y -> fzero(x -> cdf0_raw(reshape([y], 1, 1), reshape([1.0], 1, 1), x) - 0.5, 0, 2), 0, 1)
    plt!(y -> fzero(x -> cdf0_raw(reshape([y], 1, 1), reshape([1.0], 1, 1), x) - 0.9, 0, 2), 0, 1)
    plt!(y -> fzero(x -> cdf0_raw(reshape([y], 1, 1), reshape([1.0], 1, 1), x) - 0.1, 0, 2), 0, 1)
end

####################################
############### Test ###############
####################################
if true #regular GP Test
    println("Start Vanilla GP Test")
    x = x_train #reshape(x_train, length(x_train) ,1) 
    y = y_train #reshape(y_train, length(y_train), 1)
    # build and fit a GP
    #mymean = MeanLin(zeros(d))
    mymean = MeanZero() 
    # mymean = MeanZero() 
    kern = SE(0.0, 0.0) 
    gp = GP(x, y, mymean, kern) 
    optimize!(gp)     
    # predict
    μ, σ² = predict_y(gp, dropdims(x_test, dims=2)); stdv = sqrt.(σ²)
    error_GP = abs.(μ .- y_test)
    error_abs_GP = mean(error_GP)
    error_sq_GP = mean(error_GP.^2)
    CI_test_GP = hcat(-1.96.*stdv .+ μ, 1.96.*stdv .+ μ)
    count_test_GP = sum((y_test .>= CI_test_GP[:, 1]) .* (y_test .<= CI_test_GP[:,2]))/n_test
    nlpd_GP = -mean(log.(pdf.(Normal(), (y_test.-μ)./stdv)./stdv))
    println("MSE:")
    display(error_sq_GP)
    println("Conf interval accuracy:")
    display(count_test_GP)
end

#
# MSE:
# 0.0017970008992554704
# Conf interval accuracy:
# 0.941033434650456
#
#

if true
    @info "Start BTG Test"
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
            #pdf_test_i, cdf_test_i, dpdf_test_i, quantbound_test_i, support_test_i = pre_process(x_test_i, Fx_test_i, pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw)
            y_test_i = y_test[i]
            
            #median_test_i = max_train * quantile(cdf_test_i, quantbound_test_i, support_test_i)[1]
            median_test_i = fzero(x -> cdf0_raw(x_test_i, reshape([1.0], 1, 1), x) - 0.5, 0, 10)
            # @info "True, median " y_test_i_true, median_test_i
            median_set[i] = median_test_i
            try 
                #CI_test_i = max_train .* credible_interval(cdf_test_i, quantbound_test_i, support_test_i; mode=:equal, wp=.95)[1]
                aa = fzero(x -> cdf0_raw(x_test_i, reshape([1.0], 1, 1), x) - 0.975, 0, 20)
                bb = fzero(x -> cdf0_raw(x_test_i, reshape([1.0], 1, 1), x) - 0.025, 0, 20)
                CI_test_i = [bb, aa]
                count_test += (y_test_i >= CI_test_i[1])&&(y_test_i <= CI_test_i[2]) ? 1 : 0
                # @info "CI" CI_test_i
                #@info "95% confidence interval:", CI_test_i
                CI_set[i, :] = CI_test_i
            catch err
                append!(id_fail, i)
            end
            error_abs += abs(y_test_i - median_test_i)
            error_sq += (y_test_i - median_test_i)^2
            nlpd += log(pdf0_raw(y_test_i)) 
        # @info "Count, id_fail" count_test, id_fail
        catch err 
            append!(id_nonproper, i)
        end
    end
        count_test /= n_test 
        error_abs  /= n_test 
        error_sq   /= n_test 
        nlpd       /= -n_test 
        after = Dates.now()
        elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
        println("error_sq")
        display(error_sq)
        println("count_test")
        display(count_test)
        println("elapsedmin")
        display(elapsedmin)
        # Plot
        PyPlot.close("all") #close existing windows
        PyPlot.plot(x_test, median_set, label = "BTG median")
        PyPlot.plot(xs, label, label = "true")
        PyPlot.fill_between(dropdims(x_test; dims = 2), CI_set[:, 1], CI_set[:, 2], alpha = 0.3, label = "95% confidence interval")
        PyPlot.scatter(x_train, y_train, s = 10, c = "k", marker = "*")
        #PyPlot.scatter(x_test, y_test, s = 10, c = "k", marker = "*")
        PyPlot.legend(fontsize=8)
        PyPlot.grid()
        PyPlot.title("BTG $myquadtype", fontsize=10)
        PyPlot.savefig("exp_synthetic_mle_btg4.pdf")
    end