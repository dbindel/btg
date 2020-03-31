abstract type w end

mutable struct x<:w
a
end

function update(e::w)
    e.a = 5
end