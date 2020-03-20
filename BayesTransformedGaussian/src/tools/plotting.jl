"""
Generates data for plotting f on [a, b]
"""
function plt_data(f, a, b, numpts=100)
    h = (b-a)/(numpts-1)
    y = zeros(1,numpts)
    x = collect(a:h:b)
    y = map(f, x)
    return (x, y)
end

"""
Displays graph of f on interval [a, b]
"""
function plt(f, a, b, numpts=100, label = "y")
    h = (b-a)/(numpts-1)
    y = zeros(1,numpts)
    x = collect(a:h:b)
    y = map(f, x)
    display(plot(x, y, label = label))
end

function plt!(f, a, b, numpts=100, label = "y")
    h = (b-a)/(numpts-1)
    y = zeros(1,numpts)
    x = collect(a:h:b)
    y = map(f, x)
    display(plot!(x, y, label = label))
end



