set -x 

julia exp_ailerons.jl --test --single --GP --logGP > log_exp_aileron_range1.txt 2>&1

# julia exp_ailerons.jl --test --single --rangelambda -1.5 0.5 > log_exp_creep_range2.txt 2>&1

# julia exp_ailerons.jl --test --single --rangetheta 500. 3000. > log_exp_creep_range3.txt 2>&1

# julia exp_ailerons.jl --test --single ---rangetheta 500. 4000. > log_exp_creep_range4.txt 2>&1

# julia exp_ailerons.jl --test --single ---rangetheta 20. 4000. > log_exp_creep_range5.txt 2>&1
