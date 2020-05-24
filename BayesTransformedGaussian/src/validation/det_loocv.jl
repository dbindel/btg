include("../various_mle_things/incremental.jl")

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
compute determinant of submatrix of Σθ formed by deleting ith row and column efficiently
given determinant of full matrix
"""
#function det_loocv(holeskyΣθ::IncrementalCholeskyc, detcholeskyΣθ::Float64, i::Int64)
function det_loocv_IC(choleskyΣθ::IncrementalCholesky, detcholeskyΣθ::Float64, i::Int64)
    n = size(choleskyΣθ)[1]
    #println("i: ", i)
    @assert i >= 1 && n >= i
    #R = get_chol(choleskyΣθ)
    L = get_chol(choleskyΣθ).L
    return detcholeskyΣθ * norm(L\ unitvec(n, i))^2
end

"""
stably compute log determinant of submatrix of Σθ formed by deleting ith row and column efficiently
given determinant of full matrix
"""
#function det_loocv(holeskyΣθ::IncrementalCholeskyc, detcholeskyΣθ::Float64, i::Int64)
function logdet_loocv_IC(choleskyΣθ::IncrementalCholesky, logdetcholeskyΣθ::Float64, i::Int64)
    n = size(choleskyΣθ)[1]
    #println("i: ", i)
    @assert i >= 1 && n >= i
    #R = get_chol(choleskyΣθ)
    L = get_chol(choleskyΣθ).L
    return logdetcholeskyΣθ + 2*log(norm(L\ unitvec(n, i)))
end

function logdet_loocv(choleskyΣθ::Cholesky{Float64,Array{Float64,2}}, logdetcholeskyΣθ::Float64, i::Int64)
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
Compute determinant of submatrix of - [Sigmainv X; X' 0]. IC stands for IncrementalCholesky
"""
function det_XΣX_loocv_IC(X, choleskyΣθ::IncrementalCholesky, detcholeskyΣθ::Float64, i)
    n = size(choleskyΣθ, 1)
    p = size(X, 2)
    @assert n == size(X, 1)
    @assert i >=1 && n>=i 
    chol = get_chol(choleskyΣθ)
    ei = unitvec(n, i)
    #R11 = choleskyΣθ.U #upper triangular factor
    R12 = get_chol(choleskyΣθ).L\X
    cholR22 = cholesky(R12'*R12)
    a = get_chol(choleskyΣθ).L\ei
    b = - (cholR22.L \ (R12'*a))
    eWe =  norm(a)^2 - norm(b)^2
    detW = (-1)^n * detcholeskyΣθ * det(cholR22)
    det_Σθ_minus_i = det_loocv_IC(choleskyΣθ, detcholeskyΣθ, i)
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

function logdet_XΣX_loocv_IC(X, choleskyΣθ::IncrementalCholesky, logdetcholeskyΣθ::Float64, i)
    n = size(choleskyΣθ)[1]
    p = size(X, 2)
    @assert n == size(X, 1)
    @assert i >=1 && n>=i 
    ei = unitvec(n, i)
    #R11 = choleskyΣθ.U #upper triangular factor
    R12 = get_chol(choleskyΣθ).L\X
    cholR22 = cholesky(R12'*R12)
    a = get_chol(choleskyΣθ).L\ei
    b = - (cholR22.L \ (R12'*a))
    #println("norm a squared: ", norm(a)^2)
    #println("norm b squared: ", norm(b)^2)
    logeWe =  log(abs(norm(a)^2 - norm(b)^2))
    #println("logeWe: ", logeWe)
    #logdetW = (-1)^n * detcholeskyΣθ * det(cholR22)
    #println("logdet(cholR22): ", logdet(cholR22))
    logdetW = logdetcholeskyΣθ + logdet(cholR22)
    #println("logdetW: ", logdetW)
    logdet_Σθ_minus_i = logdet_loocv_IC(choleskyΣθ, logdetcholeskyΣθ, i) #submatrix of PD matrix is PD
    return logeWe + logdetW - logdet_Σθ_minus_i
end