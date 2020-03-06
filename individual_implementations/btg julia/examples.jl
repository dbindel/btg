using LinearAlgebra
include("kernel.jl")


function sample(K)
    n = size(K, 1)
    C = cholesky(K)
    z = C.L*(randn(n))
    return z
end

function getExample(dim, n, a, b, θ)
    if dim>1 
        s = [3 6 1 3; 4 2 3 2; 2 1 1 5; 1 4 2 3;5 6 7 8]
        s0 = [1 2 3 2; 2 4 2 1; 3 1 2 6; 1 9 4 2; 2 3 8 6]
        X = [3 4 4; 9 3 5; 1 7 13; 4 1 2; 5 6 14]
        X0 = [1 1 2 ; 3 2 1; 4 -1 3; 5 5 4; 8 -3 5]
        z = [9; 11; 13; 6; 7]
        example = setting(s, s0, X, X0, z)
    else 
        #s = [3 6 1 3; 4 2 3 2; 2 1 1 5; 1 4 2 3;5 6 7 8]
        #s0 = [1 2 3 2]
        s = rand(n, a)*100
        s0 = rand(1, a)*100
        X0 = rand(1, b)*100
        #X0 = [1 1 2]
        X = rand(n, b)*100
        kernelm = K(s, s, θ)
        z = sample(kernelm) 
        z = reshape(z, length(z))
        #z = [9; 11; 13; 6; 7;]
        example = setting(s, s0, X, X0, z)
    end
end
