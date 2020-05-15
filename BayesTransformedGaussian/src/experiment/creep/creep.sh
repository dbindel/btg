set -x 
julia exp_creep.jl --test --single --GP --logGP --randseed 123 > log_exp_creep_rand123.txt 2>&1
julia exp_creep.jl --test --single --GP --logGP --randseed 12 > log_exp_creep_rand12.txt 2>&1
julia exp_creep.jl --test --single --GP --logGP --randseed 13 > log_exp_creep_rand13.txt 2>&1
julia exp_creep.jl --test --single --GP --logGP --randseed 321 > log_exp_creep_rand321.txt 2>&1

julia exp_creep.jl --test --single --GP --logGP --randseed 123 --range1 1200. --range2 3000. > log_exp_creep_rand123_r2.txt 2>&1
julia exp_creep.jl --test --single --GP --logGP --randseed 12 --range1 1200. --range2 3000. > log_exp_creep_rand12_r2.txt 2>&1
julia exp_creep.jl --test --single --GP --logGP --randseed 13 --range1 1200. --range2 3000. > log_exp_creep_rand13_r2.txt 2>&1
julia exp_creep.jl --test --single --GP --logGP --randseed 321 --range1 1200. --range2 3000. > log_exp_creep_rand321_r2.txt 2>&1

julia exp_creep.jl --test --single --GP --logGP --randseed 123 --range1 1500. --range2 4000. > log_exp_creep_rand123_r3.txt 2>&1
julia exp_creep.jl --test --single --GP --logGP --randseed 12 --range1 1500. --range2 4000. > log_exp_creep_rand12_r3.txt 2>&1
julia exp_creep.jl --test --single --GP --logGP --randseed 13 --range1 1500. --range2 4000. > log_exp_creep_rand13_r3.txt 2>&1
julia exp_creep.jl --test --single --GP --logGP --randseed 321 --range1 1500. --range2 4000. > log_exp_creep_rand321_r3.txt 2>&1
