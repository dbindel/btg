
"""
define inference problem using settings
s is observed prediction locations, X is matrix of covariates, z is observed values
X0 is matrix of covariates for prediction location, s0 is prediction location
"""
struct setting{T<:Array{Float64, 2}, S<:Array{Float64, 1}}
    s::T
    s0::T
    X::T
    X0::T
    z::S
end

struct θ_params{O<:Array{Float64, 2}, C<:Cholesky{Float64,Array{Float64, 2}}}
    Eθ::O
    Eθ_prime::O
    Eθ_prime2::O
    Σθ::O
    Σθ_prime::O
    Σθ_prime2::O
    Bθ::O
    Bθ_prime::O
    Bθ_prime2::O 
    Dθ::O
    Dθ_prime::O
    Dθ_prime2::O
    Hθ::O
    Hθ_prime::O
    Hθ_prime2::O
    Cθ::O
    Cθ_prime::O
    Cθ_prime2::O
    Σθ_inv_X::O
    tripleΣ::O #X'Sigma^-1 dSigma Sigma^-1 X 
    choleskyΣθ::C
    choleskyXΣX::C
end

"""
Compute theta-dependent quantities. Unpacks setting, ...
"""
function func(θ::Float64, setting::setting{Array{Float64, 2}, Array{Float64, 1}})
    s = setting.s
    s0 = setting.s0
    X = setting.X
    X0 = setting.X0
    z = setting.z
    #zeroth order expressions and easily computed higher derivatives (building blocks)
    Eθ = fastK(s0, s0, θ, rbf_single)
    Σθ =  fastK(s, s, θ, rbf_single)
    Bθ =  fastK(s0, s, θ, rbf_single)
    choleskyΣθ = cholesky(Σθ) 
    choleskyXΣX = cholesky(Hermitian(X'*(choleskyΣθ\X))) 
    Dθ = Eθ - Bθ*(choleskyΣθ\Bθ') 
    Hθ = X0 - Bθ*(choleskyΣθ\X) 
    Cθ = Dθ + Hθ*(choleskyXΣX\Hθ') 
    Eθ_prime = K(s0, s0, θ, rbf_prime)
    Eθ_prime2 = K(s0, s0, θ, rbf_prime2)  
    Σθ_prime = fastK(s, s, θ, rbf_prime_single) 
    Σθ_prime2 = fastK(s, s, θ, rbf_prime2_single) 
    Bθ_prime = K(s0, s, θ, rbf_prime) 
    Bθ_prime2 = K(s0, s, θ, rbf_prime2) 

    #abstractions used to compute higher derivatives
    dQ = Y -> (choleskyΣθ\((Σθ_prime2*(choleskyΣθ\Y)))) - 2*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Y))))))
    Q = Y -> (choleskyΣθ\(Σθ_prime*(choleskyΣθ\(Y))))
    dPinv = Y -> choleskyXΣX\(X'*(choleskyΣθ\(Σθ_prime*(choleskyΣθ\(X*(choleskyXΣX\Y))))))
    XQX = X'*Q(X)
    dP = Y -> -XQX*Y
    d2P = Y -> -X'*(dQ(X)*Y)
    d2Pinv = Y -> -dPinv(dP(choleskyXΣX\Y)) - (choleskyXΣX\(d2P(choleskyXΣX\Y))) - (choleskyXΣX\(dP(dPinv(Y))))

    #auxiliary expressions
    Σθ_inv_X = choleskyΣθ\X
    tripleΣ = Σθ_inv_X' * Σθ_prime * Σθ_inv_X

    #higher derivatives 
    Hθ_prime = compute_Hθ_prime(Bθ_prime, Σθ_inv_X, choleskyΣθ, Σθ_prime, Bθ, X)
    Dθ_prime = compute_Dθ_prime(choleskyΣθ, Bθ, Eθ_prime, Σθ_prime, Bθ_prime)
    Cθ_prime = compute_Cθ_prime(Dθ_prime,Hθ,  Hθ_prime, choleskyXΣX, Σθ_inv_X, Σθ_prime)
    Hθ_prime2 = compute_Hθ_prime2(Bθ, Bθ_prime, Bθ_prime2, Σθ_inv_X, choleskyΣθ, X, Q, dQ)
    Dθ_prime2 = compute_Dθ_prime2(choleskyΣθ, Bθ, Bθ_prime, Bθ_prime2, Eθ_prime2, Q, dQ)
    Cθ_prime2 = compute_Cθ_prime2(Dθ_prime2, Hθ, Hθ_prime, Hθ_prime2, choleskyXΣX, dPinv, d2Pinv)
    
    x = [1 0; 0 .1]
    c = cholesky(x)

    println(typeof(Eθ))
    println(typeof(Eθ_prime))
    println(typeof(Eθ_prime2))
    println(typeof(Σθ))
    println(typeof(Σθ_prime))
    println(typeof(Σθ_prime2))
    println(typeof(Bθ))
    println(typeof(Bθ_prime))
    println(typeof(Bθ_prime2))
    println(typeof(Dθ))
    println(typeof(Dθ_prime))
    println(typeof(Dθ_prime2))
    println(typeof(Hθ))
    println(typeof(Hθ_prime))
    println(typeof(Hθ_prime2))
    println(typeof(Cθ))
    println(typeof(Cθ_prime))
    println(typeof(Cθ_prime2))
    println(typeof(Σθ_inv_X))
    println(typeof(tripleΣ))
    println(typeof(choleskyΣθ))
    println(typeof(choleskyXΣX))

    θ_params(Eθ, Eθ_prime, Eθ_prime2, Σθ, Σθ_prime, Σθ_prime2, Bθ, Bθ_prime, Bθ_prime2, Dθ, Dθ_prime, Dθ_prime2, Hθ, Hθ_prime, Hθ_prime2, Cθ, Cθ_prime, Cθ_prime2, Σθ_inv_X, tripleΣ, choleskyΣθ, choleskyXΣX)
    
    #θ_params(Eθ, Eθ_prime, Eθ_prime2, Σθ, Σθ_prime, x, Bθ, Bθ_prime, Bθ_prime2, x, Dθ_prime, x, x, Hθ_prime, x, x, x, x, x, x, c, c)
end
