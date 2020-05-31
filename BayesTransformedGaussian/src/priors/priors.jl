
abstract type priorType end

"""
Represents a uniform prior on range = [a, b]
"""
struct Uniform <:priorType 
    range::Array{Real, 2} #admissible ranges for each length scale ([lowerbound_i, upperbound_i]), stacked vertically
    d::Int64 #number of ranges in range object/number length scales
    lengths::Array{Real, 1}
    function Uniform(range) #inner constructor allows one to enforce constraints
        @assert typeof(range)<:Union{Array{T, 2}, Array{T, 1}} where T<:Real
        if length(range) > 1 
            @assert prod(range[:, 2]- range[:, 1] .> 0) #ranges must be valid
            lengths = range[:, 2] - range[:, 1]
            if typeof(range)<:Array{Real, 1}
                range = reshape(range, 1, 2)
                return new(range, 1, lengths)
            else 
                @assert Base.size(range, 2) == 2
                return new(range, Base.size(range, 1), lengths)
            end
        else
            return new(range, Base.size(range, 1), ones(size(range, 1)))
        end
    end
end

getRange(u::Uniform) = u.range #internal range array
getNum(u::Uniform) = u.d #number of dimensions of interal range array
getLengths(u::Uniform) = u.lengths
logProb(u::Uniform, x) = (@assert length(x) == u.d; l = getLengths(u); -sum(log.(l)))#density function
prob(u::Uniform, x) = (@assert length(x) == u.d; l = getLengths(u); 1/prod(l))#density function
partialx(u::Uniform, x) = 0

"""
Represents a prior on [1/b, 1/a] that induces a uniform distribution on range = [a, b]
"""
struct inverseUniform<:priorType
    range::Array
    d::Int64 #number of ranges in range object/number length scales
    lengths::Array{Real, 1}
    function inverseUniform(range)
        if length(range) > 1 
            @assert prod(range[:, 2]- range[:, 1] .> 0) #ranges must be valid
            lengths = range[:, 2] - range[:, 1]
            if typeof(range)<:Array{Real, 1}
                range = reshape(range, 1, 2)
                return new(range, 1, lengths)
            else 
                @assert Base.size(range, 2) == 2
                return new(range, Base.size(range, 1), lengths)
            end
        else
            return new(range, Base.size(range, 1), ones(size(range, 1)))
        end   
    end
end
getLengths(u::inverseUniform) = u.lengths 
prob(u::inverseUniform, x) = (@assert length(x) == u.d; l = getLengths(u); prod([1/x[i]^2 * 1 / (l[i]) for i=1:u.d]))
logProb(u::inverseUniform, x) = (@assert length(x) == u.d; l = getLengths(u); sum([-2*log(x[i]) - log(l[i]) for i=1:u.d])  )#density function

#partialx(u::inverseUniform) = -2/x^3 * 1/(u.range[2]-u.range[1])

#struct univariateExponential <:priorType
#    range::Array
#    stdev::Float64
#    mean::Float64
#    function univariateExponential(range)
#        
#    end
#end
