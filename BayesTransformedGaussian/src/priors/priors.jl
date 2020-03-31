abstract type priorType end

"""
Represents a uniform prior on range = [a, b]
"""
struct uniform <:priorType 
    range::Array
end

(u::uniform)(x) = 1/(u.range[2] - u.range[1])
partialx(u::uniform, x) = 0

"""
Represents a prior on [1/b, 1/a] that induces a uniform distribution on range = [a, b]
"""
struct inverseuniform<:priorType
    range::Array
end
(u::inverseuniform)(x) = 1/x^2 * 1/(u.range[2] - u.range[1])
partialx(u::inverseuniform) = -2/x^3 * 1/(u.range[2]-u.range[1])



#"""
#A priorInfo object contains a function handles for the prior f
#and its first and second derivatives.
#"""
#struct priorInfo{T1, T2, T3 <:Function}
#    f::T1
#    df::T2
#    d2f::T3
#end
#
#"""
#Initializes a priorInfo object given support of function
#and prior type. Currently only uniform prior is supported.
#"""
#function initialize_prior(range, type="Uniform")
#    if type=="Uniform"
#        return priorInfo(x -> 1/(range[2]-range[1]), x-> 0, x->0)
#    else
#        throw(ArgumentError("Prior type not supported."))
#    end
#end