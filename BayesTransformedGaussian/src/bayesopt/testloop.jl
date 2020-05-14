
function func()
    x = 3
    while x < 10
        x = x + 1
    end
    println(x)
end



mutable struct test
    x
end

a = test([3])

function update!(testobj, num)
    push!(testobj.x, num)
end