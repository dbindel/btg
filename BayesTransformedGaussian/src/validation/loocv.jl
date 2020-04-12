using LinearAlgebra
include("../bayesopt/incremental.jl")

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

#test problem
A = rand(5, 3) 
b = rand(5, 1)
xstar = A\b
r = b-A*xstar
aqr = qr(A)
remainders, minimizers = lsq_loocv(A, aqr, r, xstar) 
rr, mm = lsq_loocv(A, aqr, r, xstar, 3)    

remainders_actual = similar(remainders)
minimizers_actual = similar(minimizers)
for i = 1:size(A, 1)
    A1 = A[[1:i-1;i+1:end], :]
    b1 = b[[1:i-1;i+1:end]]
    x1 = A1\b1
    minimizers_actual[:, i] = x1
    remainders_actual[:, i] = b-A*x1
end
println("remainder error: ", norm(remainders_actual - remainders))
println("minimizer error: ", norm(minimizers_actual - minimizers))

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

###
### lin_sys_loocv test
###

K = [1.0         0.00369786  0.0608101;
0.00369786  1.0         0.0608101;
0.0608101   0.0608101   1.0]
println("K: ", K)
y = rand(3, 10)
#cholK = incremental_cholesky!(K, 3)

U = cholesky(K).U
c = K\y
#solve subsystem of Kc=y
c1 = lin_sys_loocv(c, U, 1)
c2 = lin_sys_loocv(c, U, 2)
c3 = lin_sys_loocv(c, U, 3)
Cs = zeros(6, 10)
for i = 1:3
    Ki = K[[1:i-1;i+1:end], [1:i-1;i+1:end]]
    Cs[2*i-1:2*i, :] = Ki\y[[1:i-1;i+1:end], :]
end
println("error of indirect error computation: ", norm(Cs - vcat(vcat(c1, c2), c3)))

cholK = incremental_cholesky!(K, 3)
c1IC = lin_sys_loocv_IC(c, cholK, 1)
c2IC = lin_sys_loocv_IC(c, cholK, 2)
c3IC = lin_sys_loocv_IC(c, cholK, 3)
println("error of indirect error computation using Incremental_Cholesky (IC): ", norm(Cs - vcat(vcat(c1IC, c2IC), c3IC)))


###
### Larger test problem. Not activated for now.
###
if true
    n = 50
    K = rand(n, n)
    K = K'*K+UniformScaling(10.0)
    y = rand(n)
    U = cholesky(K).U
    c = K\y
    #solve subsystem of Kc=y
    @time begin 
    indirectCs = zeros(n-1, n)
    for i = 1:n
        indirectCs[:, i] = lin_sys_loocv(c, U, i)
    end
    end
    @time begin
    Cs = zeros(n-1, n)
    for i = 1:n
        Ki = K[[1:i-1;i+1:end], [1:i-1;i+1:end]]
        Cs[:, i] = Ki\y[[1:i-1;i+1:end]]
    end
    end
    println("error of indirect error computation for size $n problem: ", norm(Cs - indirectCs))


    cholK = incremental_cholesky!(K, n)
    indirectCs2 = zeros(n-1, n)
    for i = 1:n
        indirectCs2[:, i] = lin_sys_loocv_IC(c, cholK, i)
    end
    println("error of indirect error computation for size $n problem using Incremental Cholesky: ", norm(Cs - indirectCs2))

end
