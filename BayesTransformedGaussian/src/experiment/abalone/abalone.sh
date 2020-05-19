set -x 
# julia exp_abalone.jl --test --single > log_exp_abalone_range1.txt 2>&1

julia exp_abalone.jl --test --single --rangelambda -0.5 0.5 > log_exp_abalone_range2.txt 2>&1

julia exp_abalone.jl --test --single --rangelambda -1.5 1. > log_exp_abalone_range3.txt 2>&1

julia exp_abalone.jl --test --single --rangelambda -1.5 1.5 > log_exp_abalone_range4.txt 2>&1
