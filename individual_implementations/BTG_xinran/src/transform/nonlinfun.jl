function BoxCox(x, lambda, IFinv = 0)
    if IFinv == 0
        if lambda == 0
            return [log(x), max(float(x).^(-1), 1e5)]
        else
            return [(expm1(lambda * log(x)))/lambda, max(float(x).^(lambda .-1), 1e5)]
        end
    else # compute the inverse function and inverse derivative
        if lambda == 0
            return [exp(x), exp(x)]
        else
            return [(lambda * x + 1)^(1/lambda), (lambda * x + 1)^(1/lambda-1)/lambda]
        end
    end
end

function ArandaOrdaz(x, lambda)
    # return function values and deivatives
    @assert 0 <= x <= 1
    if lambda == 0
        return [log(-log(1-x)),  1/(log(1-x)*(x-1))]
    else
        return [log(((1-x)^(-lambda) - 1)/lambda), (lambda * (1-x)^(-1)) / (1 - (1-x)^(lambda))]
    end
end

function DetJ(z, dg)
    n = size(z, 1)
    J = 1
    for i in 1:n
        J = J * abs(dg(z[i]))
    end
    return J
end

