# testing a set of points
# 1d case

include("model.jl")
include("plotting/plot_distribution_single.jl")

function BTG_main(training_data, testing_data)
    # training points
    x_train = training_data[1]
    z_train = training_data[2]
    n_train = size(x_train,1)

    # Test data
    x_test = testing_data[1]
    z_true = testing_data[2]
    n_test = size(x_test,1)

    for i in 1:n_test
        # select where to predict
        x_test_temp = x_test[i, :]
        z_true_temp = z_true[i]
        println("Current i = $i \n x_pred = $x_test_temp")
        flush(stdout)
        # select indices forming new training data
        distribution_temp = model(x_test_temp, training_data, kernel, nonlinfun)
        PyPlot.clf()
        plot_distribution_single(distribution_temp, x_test_temp, z_true_temp)
        PyPlot.savefig("figures/test_$i.pdf")
        println("------------------------=")
    end
    println("FINISHING PREDICTION")
end