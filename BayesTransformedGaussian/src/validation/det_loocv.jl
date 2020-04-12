include("../bayesopt/incremental.jl")

function unitvec(n, i)
    v = zeros(n, 1)
    v[i] = 1
    return v
end

"""
compute determinant of submatrix of Σθ formed by deleting ith row and column efficiently
given determinant of full matrix
"""
#function det_loocv(holeskyΣθ::IncrementalCholeskyc, detcholeskyΣθ::Float64, i::Int64)
function det_loocv(choleskyΣθ, detcholeskyΣθ::Float64, i::Int64)
    n = size(choleskyΣθ)[1]
    #println("i: ", i)
    @assert i >= 1 && n >= i
    #R = get_chol(choleskyΣθ)
    R = choleskyΣθ.U
    return detcholeskyΣθ * norm(R'\ unitvec(n, i))^2
end

"""
stably compute log determinant of submatrix of Σθ formed by deleting ith row and column efficiently
given determinant of full matrix
"""
#function det_loocv(holeskyΣθ::IncrementalCholeskyc, detcholeskyΣθ::Float64, i::Int64)
function logdet_loocv(choleskyΣθ, logdetcholeskyΣθ::Float64, i::Int64)
    n = size(choleskyΣθ)[1]
    #println("i: ", i)
    @assert i >= 1 && n >= i
    #R = get_chol(choleskyΣθ)
    R = choleskyΣθ.U
    return logdetcholeskyΣθ + 2*log(norm(R'\ unitvec(n, i)))
end

"""
Compute determinant of submatrix of - [Sigmainv X; X' 0]
"""
function det_XΣX_loocv(X, choleskyΣθ, detcholeskyΣθ::Float64, i)
    n = size(choleskyΣθ, 1)
    p = size(X, 2)
    @assert n == size(X, 1)
    @assert i >=1 && n>=i 
    ei = unitvec(n, i)
    #R11 = choleskyΣθ.U #upper triangular factor
    R12 = choleskyΣθ.L\X
    cholR22 = cholesky(R12'*R12)
    a = choleskyΣθ.L\ei
    b = - (cholR22.L \ (R12'*a))
    eWe =  norm(a)^2 - norm(b)^2
    detW = (-1)^n * detcholeskyΣθ * det(cholR22)
    det_Σθ_minus_i = det_loocv(choleskyΣθ, detcholeskyΣθ, i)
    return eWe * detW / det_Σθ_minus_i
end



"""
Stably compute log of absolute value of determinant of submatrix of - [Sigmainv X; X' 0], where
the determinant is computed up to a sign, and the log of the absolute value of 
the determinant is taken
"""
function logdet_XΣX_loocv(X, choleskyΣθ, logdetcholeskyΣθ::Float64, i)
    n = size(choleskyΣθ, 1)
    p = size(X, 2)
    @assert n == size(X, 1)
    @assert i >=1 && n>=i 
    ei = unitvec(n, i)
    #R11 = choleskyΣθ.U #upper triangular factor
    R12 = choleskyΣθ.L\X
    cholR22 = cholesky(R12'*R12)
    a = choleskyΣθ.L\ei
    b = - (cholR22.L \ (R12'*a))
    #println("norm a squared: ", norm(a)^2)
    #println("norm b squared: ", norm(b)^2)
    logeWe =  log(abs(norm(a)^2 - norm(b)^2))
    #println("logeWe: ", logeWe)
    #logdetW = (-1)^n * detcholeskyΣθ * det(cholR22)
    #println("logdet(cholR22): ", logdet(cholR22))
    logdetW = logdetcholeskyΣθ + logdet(cholR22)
    #println("logdetW: ", logdetW)
    logdet_Σθ_minus_i = logdet_loocv(choleskyΣθ, logdetcholeskyΣθ, i) #submatrix of PD matrix is PD
    return logeWe + logdetW - logdet_Σθ_minus_i
end
#####
##### test det_loocv
#####
#the inremental_cholesky is just a wrapper type that allows one to easily extend the cholesky
#decomposition without allocation more space at each iteration, because a fixed amount of space 
#is allocated at the beginning, and what you get to use is a view of the top left n x n corner

n=5
A = rand(n, n); A = A'*A + UniformScaling(1.0)
#IC = incremental_cholesky!(A, n-1) #make it n-1 just for kicks
IC = cholesky(A)
for i=1:n
    subdet = det(A[[1:i-1;i+1:end], [1:i-1;i+1:end]]) 
    println("actual subdets $i: ", subdet)
end

det_original = det(IC)
for i = 1:n
    subdet = det_loocv(IC, det_original, i)
    println("fast subdets $i: ", subdet)
end

#####
##### test det_XΣX_loocv
#####
n = 5
m = 3
X = rand(n, m)
A = (w = rand(n, n);w'*w + UniformScaling(5))
cholA = cholesky(A)
detA = det(cholA)
bigmat = [A X; X' zeros(m, m)]
#for i = 1:n
#    println("actual subdet $i: ", det(bigmat[[1:i-1;i+1:end], [1:i-1;i+1:end]]))
#end
for i = 1:n
    X_minus_i = X[[1:i-1;i+1:end], :]
    sigma_minus_i = A[[1:i-1;i+1:end], [1:i-1;i+1:end]]
     println("actual subdet $i: ", det(X_minus_i'*(sigma_minus_i\X_minus_i)))
end
for i =1:n
    subdet =  det_XΣX_loocv(X, cholA, detA, i)
    println("fast subdet $i: ", subdet)
end

#####
##### test logdet_loocv
#####
n=5
A = rand(n, n); A = A'*A + UniformScaling(1.0)
#IC = incremental_cholesky!(A, n-1) #make it n-1 just for kicks
IC = cholesky(A)
for i=1:n
    sublogdet = log(det(A[[1:i-1;i+1:end], [1:i-1;i+1:end]])) 
    println("actual sublogdets $i: ", sublogdet)
end

logdet_original = log(det(IC))
for i = 1:n
    sublogdet = logdet_loocv(IC, logdet_original, i)
    println("fast sublogdet $i: ", sublogdet)
end

#####
##### test logdet_XΣX_loocv
#####
println("Running test logdet_XΣX_loocv...")
n = 1300
m = 10
X = rand(n, m)
A = (w = rand(n, n);w'*w + UniformScaling(5))
cholA = cholesky(A)
logdetA = logdet(cholA)
bigmat = [A X; X' zeros(m, m)]
#for i = 1:n
#    println("actual subdet $i: ", det(bigmat[[1:i-1;i+1:end], [1:i-1;i+1:end]]))
#end
vals = zeros(1, n)
@time begin
for i = 1:n
    X_minus_i = X[[1:i-1;i+1:end], :]
    sigma_minus_i = A[[1:i-1;i+1:end], [1:i-1;i+1:end]]
    vals[i] = log(abs(det(X_minus_i'*(sigma_minus_i\X_minus_i))))
    #println("actual sublogdet $i: ", log(abs(det(X_minus_i'*(sigma_minus_i\X_minus_i)))))
end
end
vals1 = zeros(1, n)
@time begin
for i =1:n
    sublogdet =  logdet_XΣX_loocv(X, cholA, logdetA, i)
    vals1[i] = sublogdet
    #println("fast sublogdet $i: ", sublogdet)
end
end
println("norm of errors: ", norm(vals-vals1))