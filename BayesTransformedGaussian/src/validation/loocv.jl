using LinearAlgebra

"""
Computes remainder and minimizer for lsq problem min_x |hat(A)x-hat(b)|_2^2 where
ith row of A and b are discarded, given QR factorization of A,  
remainder r and minimizer xstar of the full problem 
"""
function lsq_loocv_single(A, AQR, r, xstar; i = 1)    
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
rr, mm = lsq_loocv_single(A, aqr, r, xstar; i = 3)    

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
loocv for linear systems
"""
function lin_sys_loocv(c, R, i)
    @assert i <= length(c)
    n = length(c)
    ve = R'\unitvec(n, i) 
    ri = c[i] / norm(ve)^2
    ci = c - ri * (R\ve)
    return ci[[1:i-1;i+1:end]] #dimension of return vector is n-1, ith entry will be zero
end


#lin_sys_loocv test

K = [1.0         0.00369786  0.0608101;
0.00369786  1.0         0.0608101;
0.0608101   0.0608101   1.0]

y = rand(3, 1)

U = cholesky(K).U
c = K\y
#solve subsystem of Kc=y
c1 = lin_sys_loocv(c, U, 1)
c2 = lin_sys_loocv(c, U, 2)
c3 = lin_sys_loocv(c, U, 3)

Cs = zeros(2, 3)
for i = 1:3
    Ki = K[[1:i-1;i+1:end], [1:i-1;i+1:end]]
    Cs[:, i] = Ki\y[[1:i-1;i+1:end]]
end
println("error of indirect error computation: ", norm(Cs - hcat(hcat(c1, c2), c3)))

n = 50
K = rand(n, n)
K = K'*K+UniformScaling(10.0)
y = rand(n, 1)

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

