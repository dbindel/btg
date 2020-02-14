# This is a 1D simple test
include("../BTG_main.jl")
include("../validation.jl")

function test_fun_1d(x)
    return sin(x) + 1.1
end

# training points
x_train = [-4, -3, -2, 0, 1, 2, 3, 4]
y_train = test_fun_1d.(x_train) 
n_train = size(x_train,1)
training_data = [x_train, y_train]

# Test data
x_test = range(-5, stop = 5, step = 0.23)
y_test = test_fun_1d.(x_test) 
testing_data = [x_test, y_test];

# model setting
kernel = Kernel_SE
nonlinfun = BoxCox

# cross validation
print("STARTING CROSS VALIDATION \n")
cross_validation(training_data, kernel, nonlinfun)

# call the BTG model 
print("STARTING PREDICTION \n")
BTG_main(training_data, testing_data)

