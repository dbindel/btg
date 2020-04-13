
function func2(x)
return x^2
end

function func1(x)
    function func3(x)
        return func4(x)
    end
return func3(x)
end

function anotherfun(x)
    return awayfun(x)
end

function nonanonfun(x)
    return anotherfun(x)
end

anonfun = (x::Float64) -> nonanonfun(x)
anonfun = (x::Int64) -> nonanonfun(x)

