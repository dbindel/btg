#!/bin/bash
set -x

# gaussian + gaussian, single
julia exp_abalone.jl --test --single --quadtype 1  > log_time_quad1_s_posc7_200.txt 2>&1
julia exp_abalone.jl --test --single --quadtype 1  --ntrain 400 > log_time_quad1_s_posc7_400.txt 2>&1
julia exp_abalone.jl --test --single --quadtype 1  --ntrain 1000 > log_time_quad1_s_posc7_1000.txt 2>&1

# sparsegrid + Gaussian, multi
julia exp_abalone.jl --test --quadtype 2  --ntrain 200 > log_time_quad2_s_posc7_200.txt 2>&1
julia exp_abalone.jl --test --quadtype 2  --ntrain 400 > log_time_quad2_s_posc7_400.txt 2>&1
julia exp_abalone.jl --test --quadtype 2  --ntrain 1000 > log_time_quad2_s_posc7_1000.txt 2>&1

# sparse grid, whole, multi 
julia exp_abalone.jl --test --quadtype 3  --ntrain 200 > log_time_quad3_s_posc7_200.txt 2>&1
julia exp_abalone.jl --test --quadtype 3  --ntrain 400 > log_time_quad3_s_posc7_400.txt 2>&1
julia exp_abalone.jl --test --quadtype 3  --ntrain 1000 > log_time_quad3_s_posc7_1000.txt 2>&1

# QMC, multi
julia exp_abalone.jl --test --quadtype 4 --ntrain 200 > log_time_quad4_s_posc7_200.txt 2>&1
julia exp_abalone.jl --test --quadtype 4 --ntrain 400 > log_time_quad4_s_posc7_400.txt 2>&1
julia exp_abalone.jl --test --quadtype 4 --ntrain 1000 > log_time_quad4_s_posc7_1000.txt 2>&1
