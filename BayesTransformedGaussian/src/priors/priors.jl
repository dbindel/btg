abstract type priorType end
abstract type Uni

"""
Represents a uniform prior on range = [a, b]
"""
struct Uniform <:priorType 
    range::Array{Real, 2} #admissible ranges for each length scale ([lowerbound_i, upperbound_i]), stacked vertically
    d::Int64 #number of ranges in range object/number length scales
    lengths::Array{Real, 1}
    function Uniform(range) #inner constructor allows one to enforce constraints
        @assert typeof(range)<:Union{Array{T, 2}, Array{T, 1}} where T<:Real
        @assert prod(range[:, 2]- range[:, 1] .> 0) #ranges must be valid
        lengths = range[:, 2] - range[:, 1]
        if typeof(range)<:Array{Real, 1}
            range = reshape(range, 1, 2)
            return new(range, 1, lengths)
        else 
            @assert size(range, 2) == 2
            return new(range, size(range, 1), lengths)
        end
    end
end

getRange(u::Uniform) = u.range #internal range array
getNum(u::Uniform) = u.d #number of dimensions of interal range array
getLengths(u::Uniform) = u.lengths

(u::Uniform)(x) = (@assert length(x) == u.d; l = getLength(u); 1/prod(l))#density function

partialx(u::Uniform, x) = 0

"""
Represents a prior on [1/b, 1/a] that induces a uniform distribution on range = [a, b]
"""
struct inverseuniform<:priorType
    range::Array
end
(u::inverseuniform)(x) = 1/x^2 * 1/(u.range[2] - u.range[1])
partialx(u::inverseuniform) = -2/x^3 * 1/(u.range[2]-u.range[1])
