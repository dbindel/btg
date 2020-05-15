set -x

# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 1  > log_sythetic_range_5_20_noise1.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 2  > log_sythetic_range_5_20_noise2.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 3  > log_sythetic_range_5_20_noise3.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 4  > log_sythetic_range_5_20_noise4.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 5  > log_sythetic_range_5_20_noise5.txt 2>&1

# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 1  > log_sythetic_range_3_20_noise1.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 2  > log_sythetic_range_3_20_noise2.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 3  > log_sythetic_range_3_20_noise3.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 4  > log_sythetic_range_3_20_noise4.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 5  > log_sythetic_range_3_20_noise5.txt 2>&1

# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 1  > log_sythetic_range_4_20_noise1.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 2  > log_sythetic_range_4_20_noise2.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 3  > log_sythetic_range_4_20_noise3.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 4  > log_sythetic_range_4_20_noise4.txt 2>&1
# julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 5  > log_sythetic_range_4_20_noise5.txt 2>&1

julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 1 --p 2 > log_sythetic_range_5_20_noise1_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 2 --p 2 > log_sythetic_range_5_20_noise2_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 3 --p 2 > log_sythetic_range_5_20_noise3_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 4 --p 2 > log_sythetic_range_5_20_noise4_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 5 --p 2 > log_sythetic_range_5_20_noise5_p2.txt 2>&1

julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 1 --p 2  > log_sythetic_range_3_20_noise1_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 2 --p 2  > log_sythetic_range_3_20_noise2_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 3 --p 2  > log_sythetic_range_3_20_noise3_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 4 --p 2  > log_sythetic_range_3_20_noise4_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 5 --p 2  > log_sythetic_range_3_20_noise5_p2.txt 2>&1

julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 1 --p 2  > log_sythetic_range_4_20_noise1_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 2 --p 2  > log_sythetic_range_4_20_noise2_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 3 --p 2  > log_sythetic_range_4_20_noise3_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 4 --p 2  > log_sythetic_range_4_20_noise4_p2.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 5 --p 2  > log_sythetic_range_4_20_noise5_p2.txt 2>&1




julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 1 --randseed 123 > log_sythetic_range_5_20_noise1_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 2 --randseed 123 > log_sythetic_range_5_20_noise2_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 3 --randseed 123 > log_sythetic_range_5_20_noise3_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 4 --randseed 123 > log_sythetic_range_5_20_noise4_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 5 --randseed 123 > log_sythetic_range_5_20_noise5_p1_rand123.txt 2>&1

julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 1 --randseed 123 > log_sythetic_range_3_20_noise1_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 2 --randseed 123 > log_sythetic_range_3_20_noise2_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 3 --randseed 123 > log_sythetic_range_3_20_noise3_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 4 --randseed 123 > log_sythetic_range_3_20_noise4_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 5 --randseed 123 > log_sythetic_range_3_20_noise5_p1_rand123.txt 2>&1

julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 1 --randseed 123 > log_sythetic_range_4_20_noise1_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 2 --randseed 123 > log_sythetic_range_4_20_noise2_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 3 --randseed 123 > log_sythetic_range_4_20_noise3_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 4 --randseed 123 > log_sythetic_range_4_20_noise4_p1_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 5 --randseed 123  > log_sythetic_range_4_20_noise5_p1_rand123.txt 2>&1





julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 1 --randseed 123 --p 2 > log_sythetic_range_5_20_noise1_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 2 --randseed 123 --p 2 > log_sythetic_range_5_20_noise2_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 3 --randseed 123 --p 2 > log_sythetic_range_5_20_noise3_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 4 --randseed 123 --p 2 > log_sythetic_range_5_20_noise4_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.5 --lmax 2. --noiselevel 5 --randseed 123 --p 2 > log_sythetic_range_5_20_noise5_p2_rand123.txt 2>&1

julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 1 --randseed 123 --p 2  > log_sythetic_range_3_20_noise1_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 2 --randseed 123 --p 2  > log_sythetic_range_3_20_noise2_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 3 --randseed 123 --p 2  > log_sythetic_range_3_20_noise3_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 4 --randseed 123 --p 2  > log_sythetic_range_3_20_noise4_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.3 --lmax 2. --noiselevel 5 --randseed 123 --p 2  > log_sythetic_range_3_20_noise5_p2_rand123.txt 2>&1

julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 1 --randseed 123 --p 2  > log_sythetic_range_4_20_noise1_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 2 --randseed 123 --p 2  > log_sythetic_range_4_20_noise2_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 3 --randseed 123 --p 2  > log_sythetic_range_4_20_noise3_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 4 --randseed 123 --p 2  > log_sythetic_range_4_20_noise4_p2_rand123.txt 2>&1
julia exp_synthetic1.jl --test --GP --logGP --lmin 0.4 --lmax 2. --noiselevel 5 --randseed 123 --p 2  > log_sythetic_range_4_20_noise5_p2_rand123.txt 2>&1


