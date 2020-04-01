abstract type priorType end

"""
Represents a uniform prior on range = [a, b]
"""
struct Uniform <:priorType 
    range::Array
end

(u::Uniform)(x) = 1/(u.range[2] - u.range[1])
partialx(u::uniform, x) = 0

"""
Represents a prior on [1/b, 1/a] that induces a uniform distribution on range = [a, b]
"""
struct inverseuniform<:priorType
    range::Array
end
(u::inverseuniform)(x) = 1/x^2 * 1/(u.range[2] - u.range[1])
partialx(u::inverseuniform) = -2/x^3 * 1/(u.range[2]-u.range[1])
