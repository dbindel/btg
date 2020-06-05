using GaussianProcesses
using Random

Random.seed!(20140430)
# Training data
n=183;                          #number of training points
x = 2π * rand(n);              #predictors
y = sin.(x) + 0.05*randn(n);   #regressors

#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
                    # log standard deviation of observation noise (this is optional)
gp = GP(x,y,mZero,kern)       #Fit the GP