"""
A priorInfo object contains a function handles for the prior f
and its first and second derivatives.
"""
struct priorInfo{T1, T2, T3 <:Function}
    f::T1
    df::T2
    d2f::T3
end

"""
Initializes a priorInfo object given support of function
and prior type. Currently only uniform prior is supported.
"""
function initialize_prior(range, type="Uniform")
    if type=="Uniform"
        return priorInfo(x -> 1/(range[2]-range[1]), x-> 0, x->0)
    else
        throw(ArgumentError("Prior type not supported."))
    end
end