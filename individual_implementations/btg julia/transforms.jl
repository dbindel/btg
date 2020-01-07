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
    lambda==0 ? x.^(-1) : x.^(lambda-1)
end

"""
Inverse Box Cox power transformation
"""
function invBoxCox(y, lambda=1)
    lambda==0 ? Base.exp.(y) : Base.exp.(Base.log.(lambda.*y.+1)./lambda)
end


