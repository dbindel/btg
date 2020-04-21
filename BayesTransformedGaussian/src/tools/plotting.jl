#import Plots
import PyPlot

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
function plt(f, a, b, numpts=100; label = "y", title="")
    h = (b-a)/(numpts-1)
    y = zeros(1,numpts)
    x = collect(a:h:b)
    y = map(f, x)
    display(Plots.plot(x, y, label = label, title = title))
    return (x, y)
end

function plt!(f, a, b, numpts=100; label = "y", title = "")
    h = (b-a)/(numpts-1)
    y = zeros(1,numpts)
    x = collect(a:h:b)
    y = map(f, x)
    display(Plots.plot!(x, y, label = label, title = title))
    return(x, y)
end

function plt!(point::Float64; label = "", title = "")
    display(Plots.plot!([point], seriestype = :vline))
end
