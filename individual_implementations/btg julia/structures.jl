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


struct θ_params{T<:Array{Float64, 2}, C<:Cholesky{Float64,Array{Float64, 2}}}
    Eθ::T
    Σθ::T
    Bθ::T
    Dθ::T
    Hθ::T
    Cθ::T
    Eθ_prime::T
    Eθ_prime2::T
    Σθ_prime::T
    Σθ_prime2::T
    Bθ_prime::T
    Bθ_prime2::T 
    choleskyΣθ::C
    choleskyXΣX::C
end

