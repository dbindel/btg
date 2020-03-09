struct composite{T<:Array{Float64}, S<:Float64}
    a::T
    b::S
end

struct composite2{T<:Array{Float64}, S<:Int64}
    a::T
    b::S
end

obj = composite([1.0;2;3], 5.6)
obj2 = composite2([1.0;3], 5)
function gg(x, y::Union{Int64, Float64, Nothing}=nothing)
    if y==nothing
        return x^2
    else
        return x
end
end