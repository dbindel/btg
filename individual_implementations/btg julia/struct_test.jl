struct composite{T<:Array{Float64}, S<:Float64}
    a::T
    b::S
end

function gg(x, y::Union{Int64, Float64, Nothing}=nothing)
    if y==nothing
        return x^2
    else
        return x
end
end