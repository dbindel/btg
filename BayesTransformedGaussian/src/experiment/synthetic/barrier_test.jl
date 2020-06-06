using GaussianProcesses
using Plots

include("barrier_btg.jl")
include("../../computation/tdist_mle.jl")
include("../../computation/opt_tdist_mle.jl")

btg0 = load_synthetic_btg()

println("lambda nodes")
display(sort(collect((keys(btg0.λbuffer_dict))))')

randseed = getseed()
Random.seed!(randseed);

#btg0 = btg(trainingData0, rangeθ, rangeλ; quadtype = myquadtype)
(pdf0_raw, cdf0_raw, dpdf0_raw, quantInfo0_raw) = solve(btg0);

######
###### Handy test code snippets
######

#v(x) = cdf0_raw(reshape([0.05], 1, 1), reshape([1.0], 1, 1), x)


f = theta -> tdist_mle(btg0, theta, -0.31175)
g(theta, lambda) = tdist_mle(btg0, theta, lambda)
g_vec(arr) = tdist_mle(btg0, abs(arr[1]), arr[2])
h = lambda -> tdist_mle(btg0, 140.0, lambda)
h_boxed = (lambda_boxed) -> tdist_mle(btg0, 140.0, lambda_boxed[1])

MLE_ESTIMATION = false
if MLE_ESTIMATION
    lambda_star = optimize_lambda(h_boxed, [-3, 3])
    println("lambda_star")
    display(lambda_star)

    opt_vec = optimize_theta_lambda_single(g_vec, [1.0, 1000.0], [-5.0, 5.0])
    println("optimal theta-lambda pair")
    display(opt_vec)
end

#barrier(x) = 0.05/(0.001+(x-1)^2)  + 0.05/(0.001+(x-2)^2) - log(1-2(x/2-2)) + 1 + sin(x+exp(x))/10 + exp(-(x-2)^2)
#Plots.plot()
#plt(barrier, 0, 4.999)

if true #regular GP Test
    println("Start Vanilla GP Test")
    x = x_train #reshape(x_train, length(x_train) ,1) 
    y = y_train #reshape(y_train, length(y_train), 1)
    # build and fit a GP
    #mymean = MeanLin(zeros(d))
    mymean = MeanZero() 
    # mymean = MeanZero() 
    kern = SE(0.0, 0.0) 
    gp = GP(dropdims(x, dims=2), y, mymean, kern) 
    GaussianProcesses.optimize!(gp)     
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
####################################
############### Test ###############
####################################

#TODO:
#scatter training and testing points
#fine mesh for true mean and btg mean
UB = parsed_args["upper_bound_estimate"]
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
    #summary statistics loop
    for i in 1:n_test
        global error_abs, error_sq, nlpd, count_test
        # mod(i, 20) == 0 ? (@info i) : nothing
        # @info "i" i
        x_test_i = reshape(x_test[i, :], 1, d)
        Fx_test_i = reshape(Fx_test[i, :], 1, p)
        try
            y_test_i = y_test[i]
            
            median_test_i = fzero(x -> cdf0_raw(x_test_i, reshape([1.0], 1, 1), x) - 0.5, 0, UB)
            median_set[i] = median_test_i
            try #compute confidence interval bounds
                aa = fzero(x -> cdf0_raw(x_test_i, reshape([1.0], 1, 1), x) - 0.975, 0, UB) 
                bb = fzero(x -> cdf0_raw(x_test_i, reshape([1.0], 1, 1), x) - 0.025, 0, UB)
                CI_test_i = [bb, aa]
                count_test += (y_test_i >= CI_test_i[1])&&(y_test_i <= CI_test_i[2]) ? 1 : 0
                CI_set[i, :] = CI_test_i
            catch err
                append!(id_fail, i)
            end
            error_abs += abs(y_test_i - median_test_i)
            error_sq += (y_test_i - median_test_i)^2
            nlpd += log(pdf0_raw(y_test_i)) 
        catch err 
            append!(id_nonproper, i)
        end
    end

        #compute summary statitics
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

        #plotting loop for high res btg mean/conf intervals and true mean
        N_plot = parsed_args["N_plot"] 
        plot_CI_set = zeros(N_plot, 2)
        plot_median_set = zeros(N_plot)
        mesh_x = range(xs[1], stop = xs[end], length = N_plot) 
        mesh_y = fun.(mesh_x)
        count_test_plot = 0.0
        for i in 1:N_plot
            global error_abs, error_sq, nlpd, count_test_plot
            # mod(i, 20) == 0 ? (@info i) : nothing
            # @info "i" i
            x_test_i = reshape([mesh_x[i]], 1, d)
            Fx_test_i = covariate_fun(x_test_i, 1) #1 indicates constant mean 
            try
                y_test_i = fun(x_test_i[1])
                median_test_i = fzero(x -> cdf0_raw(x_test_i, reshape([1.0], 1, 1), x) - 0.5, 0, UB)
                plot_median_set[i] = median_test_i
                try #compute confidence interval bounds
                    aa = fzero(x -> cdf0_raw(x_test_i, reshape([1.0], 1, 1), x) - 0.975, 0, UB) 
                    bb = fzero(x -> cdf0_raw(x_test_i, reshape([1.0], 1, 1), x) - 0.025, 0, UB)
                    CI_test_i = [bb, aa]
                    count_test_plot += (y_test_i >= CI_test_i[1])&&(y_test_i <= CI_test_i[2]) ? 1 : 0
                    plot_CI_set[i, :] = CI_test_i
                catch err
                    append!(id_fail, i)
                end
                error_abs += abs(y_test_i - median_test_i)
                error_sq += (y_test_i - median_test_i)^2
                nlpd += log(pdf0_raw(y_test_i)) 
            catch err 
                append!(id_nonproper, i)
            end
        end
        # Plot true mean and btg mean
        PyPlot.close("all") #close existing windows
        if parsed_args["log_scale"]
            PyPlot.yscale("log")
        end
        PyPlot.plot(mesh_x, plot_median_set, label = "BTG median")
        PyPlot.plot(mesh_x, mesh_y, label = "true")
        PyPlot.fill_between(mesh_x, plot_CI_set[:, 1], plot_CI_set[:, 2], alpha = 0.3, label = "95% confidence interval")
        PyPlot.scatter(x_train, y_train, s = 30, color = "blue", marker = "*")
        PyPlot.scatter(x_test, y_test, s = 30, color = "red", marker = "o")
        PyPlot.legend(fontsize=8)
        PyPlot.grid()
        PyPlot.title("BTG $myquadtype", fontsize=10)
        #PyPlot.savefig("exp_synthetic_mle_btg4.pdf")
    end