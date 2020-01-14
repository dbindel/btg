"""
...
# Arguments
* `x::float64`: evaluation point
* `lambda::float=1.0`: lambda hyperparameter
"""
function boxCox(x, lambda=1)
    lambda == 0 ? Base.log.(x) : (x.^lambda.-1)./lambda
end

"""
Derivative of Box-Cox power transformation w.r.t x
"""
function boxCoxPrime(x, lambda=1)
    lambda==0 ? x.^(-1) : x.^(lambda .-1)
end

"""
Second derivative of Box-Cox power transformation w.r.t x
"""
function boxCoxPrime2(x, lambda=1)
    lambda==0 ? -x.^(-2) : (lambda-1)*x.^(lambda-2)
end

"""
Derivative of Box-Cox power transformation w.r.t lambda
"""
function boxCoxPrime_lambda(x, lambda=1)
    lambda==0 ? 0 : (lambda * x.^lambda .* log.(x) .- x.^lambda .+ 1)/lambda^2
end

"""
Second derivative of Box-Cox power transformation w.r.t lambda
"""
function boxCoxPrime_lambda2(x, lambda=1)
    lambda==0 ? 0 : x.^lambda .* (Base.log.(x)^2)/lambda - 
    (2*x.^lambda .* Base.log.(x))/lambda^2 + 2*(x.^lambda-1)/lambda^3
end

"""
Mixed derivative of Box-Cox power transformation w.r.t lambda and x
"""
function boxCoxMixed_lambda_z(x, lambda=1)
    lambda==0 ? 0 : x.^(lambda-1) .* Base.log.(x) 
end

"""
Inverse Box Cox power transformation
"""
function invBoxCox(y, lambda=1)
    lambda==0 ? Base.exp.(y) : Base.exp.(Base.log.(lambda.*y.+1)./lambda)
end


