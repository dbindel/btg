# testing a set of points
# 1d case
using Cubature

include("model.jl")
include("plotting/plot_distribution_single.jl")
include("root_finding/zero_finding.jl")

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
        # single point prediction
        start = time()
        distribution_temp = model(x_test_temp, training_data, kernel, nonlinfun)
        elapsed = time() - start
        print("time taken for single point prediction: $elapsed \n")

        # check pdf integral == 1
        # get the current pdf
        pdf_temp = distribution_temp.pdf
        # get the upper bound for integral
        b = distribution_temp.upperbd
        int_temp = hquadrature(pdf_temp, 1e-6, b)[1]
        print("Current integral interval: [0, $b], int(pdf) = $int_temp \n")

        # plot 
        PyPlot.clf()
        plot_distribution_single(distribution_temp, x_test_temp, z_true_temp)
        PyPlot.savefig("figures/test_$i.pdf")
        println("------------------------")
    end
end