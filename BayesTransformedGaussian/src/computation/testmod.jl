module testmod

export hi, func1

mutable struct hi
a::Int64
end

function func1(x::hi)
    return x.a^2
end

end