using LinearAlgebra

"""
LOOCV for least squares problem min_x |Ax-b|_2^2

"""
function loocv(A, AQR, r, xstar)    
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
        println("gamma: ", gamma)
    end
    return remainders, minimizers
end

#test problem
A = rand(5, 3) 
b = rand(5, 1)
xstar = A\b
r = b-A*xstar
aqr = qr(A)
remainders, minimizers = loocv(A, aqr, r, xstar) 

#
A1 = A[2:end, :]
b1 = b[2:end]
sol = A1\b1
