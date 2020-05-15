set -x

julia exp_abalone.jl --test --singletest --ntrain 400 --ntest 50 --sparse 

julia exp_abalone.jl --test --singletest --ntrain 400 --ntest 50 --single

julia exp_abalone.jl --test --singletest --ntrain 400 --ntest 50 

julia exp_abalone.jl --test --singletest --ntrain 800 --ntest 50 --sparse --single

julia exp_abalone.jl --test --singletest --ntrain 800 --ntest 50 --sparse 

julia exp_abalone.jl --test --singletest --ntrain 800 --ntest 50 --single

julia exp_abalone.jl --test --singletest --ntrain 800 --ntest 50 

julia exp_abalone.jl --test --singletest --ntrain 1000 --ntest 50 --sparse --single

julia exp_abalone.jl --test --singletest --ntrain 1000 --ntest 50 --sparse 

julia exp_abalone.jl --test --singletest --ntrain 1000 --ntest 50 --single

julia exp_abalone.jl --test --singletest --ntrain 1000 --ntest 50 


