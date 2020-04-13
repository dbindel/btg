

##module mod2 
include("testmod.jl")

#import testmod: hi1, func1
using .testmod

mutable struct hi2
a::Int64
end

function func2(x::hi2)
    return x.a^3
end

#end