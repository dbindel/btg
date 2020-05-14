# prediction on abalone, 
# Gaussian quadrature for both parameter + single lengthscale
# choose number of training data in [1, 1000]
# choose number of testing data in [1, 3176]
# dafault: ntrain = 200, ntest = 100
# error history and prediction results are stored separately

julia profile_abalone.jl --test --ntrain 500  --ntest 50