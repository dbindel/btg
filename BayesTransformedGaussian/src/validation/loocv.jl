using LinearAlgebra
include("../various_mle_things/incremental.jl")

"""
Computes remainder and minimizer for lsq problem min_x |hat(A)x-hat(b)|_2^2 where
ith row of A and b are discarded, given QR factorization of A,  
remainder r and minimizer xstar of the full problem 
"""
function lsq_loocv(A, AQR, r, xstar, i)    
    n = size(A, 1)
    p = size(A, 2)
    @assert n>=p #more rows than columns
    @assert i>=1 && n>=i #i must be valid row
    Q = AQR.Q[1:n, 1:p] #get reduced QR factor
    R = AQR.R
    At = A'
    r = reshape(r, length(r), 1) #reshape into col vector
    xstar = reshape(xstar, length(xstar), 1) #reshape into col vector
    remainder = repeat(r, 1, 1) #initial value (r) is important to algorithm
    minimizer = repeat(xstar, 1, 1) #initial value (xstar) is important to algorithm
    rmv = @views remainder
    minv = @views minimizer
    Atv = @views At 
    leverage_score = norm(Q[i, :])^2 #norm of ith row of Q squared
    gamma = r[i] / (1-leverage_score)   
    minv[:, 1] .-= R\(R'\ Atv[:, i]) * gamma
    rmv[:, 1] .+= Q * (Q'[:, i]) * gamma
    return remainder, minimizer #col vectors
end

"""
LOOCV for least squares problem min_x |Ax-b|_2^2. Computes all n remainder and minimizer vectors. Primarily used for testing.
"""
function lsq_loocv(A, AQR, r, xstar)    
    n = size(A, 1)
    p = size(A, 2)
    @assert n>=p #more rows than columns
    Q = AQR.Q[1:n, 1:p] #get reduced QR factor
    R = AQR.R
    At = A'
    r = reshape(r, length(r), 1) #reshape into col vector
    xstar = reshape(xstar, length(xstar), 1) #reshape into col vector
    remainders = repeat(r, 1, n) #length n b/c we have n subproblems
    minimizers = repeat(xstar, 1, n)
    rmv = @views remainders
    minv = @views minimizers
    Atv = @views At 
    for i = 1:n
        leverage_score = norm(Q[i, :])^2 #norm of ith row of Q squared
        gamma = r[i] / (1-leverage_score)   
        minv[:, i] .-= R\(R'\ Atv[:, i]) * gamma
        rmv[:, i] .+= Q * (Q'[:, i]) * gamma
    end
    return remainders, minimizers
end


function unitvec(n, i)
    v = zeros(n, 1)
    v[i] = 1
    return v
end

"""
loocv for linear system Kc = y, generalized to case where c is a matrix
    INPUTS: R: upper triangular Cholesky factor of K
            c: solution to Kc = y
            i: row up for deletion
"""
function lin_sys_loocv(c, R, i)
    n = size(R)[1]
    #println(n)
    if size(c, 2)==1
        c = reshape(c, length(c), 1)
    end
    @assert i <= size(c, 1)
    ret = Array{Float64, 2}(undef, size(c, 1) - 1, size(c, 2)) 
    for j = 1:size(ret, 2)
        #println(unitvec(n, i) )
        #println(R')
        ve = R'\ unitvec(n, i) 
        ri = c[i, j] / norm(ve)^2
        ci = c[:, j] - ri * (R\ve)
        ret[:, j] = ci[[1:i-1;i+1:end]] #dimension of return vector is n-1, ith entry will be zero
    end
    return ret
end

"""
Use incremental cholesky in computation
"""
function lin_sys_loocv_IC(c::Array{T} where T, IC::IncrementalCholesky{Float64}, i::Int64)
    R = get_chol(IC).U
    L = get_chol(IC).L
    n = size(IC)[1]
    if size(c, 2)==1
        c = reshape(c, length(c), 1)
    end
    @assert i <= size(c, 1)
    ret = Array{Float64, 2}(undef, size(c, 1) - 1, size(c, 2)) 
    for j = 1:size(ret, 2)
        #println(unitvec(n, i) )
        #println(R')
        #ve = R'\ unitvec(n, i) 
        ve = L\unitvec(n, i) 
        ri = c[i, j] / norm(ve)^2
        ci = c[:, j] - ri * (R\ve)
        ret[:, j] = ci[[1:i-1;i+1:end]] #dimension of return vector is n-1, ith entry will be zero
    end
    return ret
end

