#!/bin/bash
set -x

julia exp_abalone.jl --test --sparse --single > log_abalone_sparse_s_posc3.txt 2>&1

julia exp_abalone.jl --test --sparse > log_abalone_sparse_m_posc3.txt 2>&1

julia exp_abalone.jl --test --single > log_abalone_qmc_s_posc3.txt 2>&1

julia exp_abalone.jl --test > log_abalone_qmc_m_posc3.txt 2>&1

julia exp_abalone.jl --test --sparse --single --posc 7 > log_abalone_sparse_s_posc7.txt 2>&1

julia exp_abalone.jl --test --sparse --posc 7 > log_abalone_sparse_m_posc7.txt 2>&1

julia exp_abalone.jl --test --single --posc 7 > log_abalone_qmc_s_posc7.txt 2>&1

julia exp_abalone.jl --test --posc 7 > log_abalone_qmc_m_posc7.txt 2>&1

julia exp_creep.jl --test --sparse --single > log_creep_sparse_s_posc30.txt 2>&1

julia exp_creep.jl --test --single > log_creep_qmc_s_posc30.txt 2>&1

julia exp_creep.jl --test > log_creep_qmc_m_posc30.txt 2>&1


