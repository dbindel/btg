struct priorInfo{T1, T2, T3 <:Function}
    f::T1
    df::T2
    d2f::T3
end

function initialize_prior(range, type="Uniform")
    if type=="Uniform"
        return priorInfo(x -> 1/(range[2]-range[1]), x-> 0, x->0)
    else
        throw(ArgumentError("Prior type not supported."))
    end
end