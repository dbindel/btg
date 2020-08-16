set -x 
# julia exp_abalone.jl --test --single > log_exp_abalone_range1.txt 2>&1

# julia exp_abalone.jl --test --single --rangelambda -0.5 0.5 > log_exp_abalone_range2.txt 2>&1

# julia exp_abalone.jl --test --single --rangelambda -1.5 1. > log_exp_abalone_range3.txt 2>&1

# julia exp_abalone.jl --test --single --rangelambda -1.5 1.5 > log_exp_abalone_range4.txt 2>&1

# May 27
# try identity transform
# julia exp_abalone.jl --test --single 2>&1 | tee log_exp_abalone_range1.txt 

# # try constant covariate
# julia exp_abalone.jl --test --single --posc 0 2>&1 | tee log_exp_abalone_range1.txt 

# May 28
# try ShiftedBoxCox with more postive lambda range
# julia exp_abalone.jl --test --single --transform "ShiftedBoxCox" 2>&1 | tee log_exp_abalone_shiftedBoxCox.txt

# May 30 
# use inverse uniform distribution of theta
# julia exp_abalone.jl --test --single --randseed 123 2>&1 | tee log_exp_abalone_inverseuniform_rand123.txt
# julia exp_abalone.jl --test --single --transform "IdentityTransform" --randseed 123 2>&1 | tee log_exp_abalone_inverseuniform_idtrans_rand123.txt

# julia exp_abalone.jl --test --single 2>&1 | tee log_exp_abalone_inverseuniform.txt
# julia exp_abalone.jl --test --single --transform "IdentityTransform" 2>&1 | tee log_exp_abalone_inverseuniform_idtrans.txt

# Aug 16
julia exp_abalone.jl --test --single --ntrain 100 --ntest 5 --posc 3 2>&1 |tee log_exp_abalone_test.txt 
