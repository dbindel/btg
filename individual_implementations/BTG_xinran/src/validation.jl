include("model.jl")
include("plotting/plot_distribution_single.jl")
using PyPlot

function cross_validation(training_data, kernel, nonlinfun)
    x_train = training_data[1]
    z_train = training_data[2]
    n_train = length(z_train)
    for i in 1:n_train
        # select where to predict
        x_test = x_train[i, :]
        z_true = z_train[i]
        println("Current i = $i \n x_pred = $x_test")
        flush(stdout)
        # select indices forming new training data
        if i == 1
            idx = collect(i+1:n_train)
        elseif i == n_train
            idx = collect(1:i-1)
        else
            idx = vcat(collect(1:i-1),collect(i+1:n_train))
        end
        x_train_temp = x_train[idx]
        z_train_temp = z_train[idx]
        training_data_temp = [x_train_temp, z_train_temp]
        distribution_temp = model(x_test, training_data_temp, kernel, nonlinfun)
        PyPlot.clf()
        plot_distribution_single(distribution_temp, x_test, z_true)
        PyPlot.savefig("figures/cross_validation_$i.pdf")
        println("------------------------")
    end
    println("FINISHING CROSS VALIDATION")

end