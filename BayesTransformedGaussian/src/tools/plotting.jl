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
function plt(f, a, b, numpts=100; label = "y", title="", color = nothing)
    h = (b-a)/(numpts-1)
    y = zeros(1,numpts)
    x = collect(a:h:b)
    y = map(f, x)
    if color!=nothing
        display(Plots.plot!(x, y, label = label, title = title, color = color, linewidth = 3))
    else 
        display(Plots.plot!(x, y, label = label, title = title, linewidth = 3))
    end
    return (x, y)
end

function plt!(f, a, b, numpts=100; label = "y", title = "", color = nothing)
    h = (b-a)/(numpts-1)
    y = zeros(1,numpts)
    x = collect(a:h:b)
    y = map(f, x)
    if color!=nothing
        display(Plots.plot!(x, y, label = label, title = title, color = color, linewidth = 3))
    else 
        display(Plots.plot!(x, y, label = label, title = title, linewidth = 3))
    end
    return(x, y)
end

function plt!(point::Float64; label = "", title = "")
    display(Plots.plot!([point], seriestype = :vline))
end



"""
plot pdf, median, mode and credible interval
"""
function plot_distribution(pdf, median, y0_true; CI=nothing, mytitle="Posterior Distribution")
    # plot pdf
    b = 1.2
    while pdf(b) > 0.05 && b < 10
        b += 0.5 
    end
    xgrid = range(1e-2, stop=b, length=100)
    ygrid = pdf.(xgrid);
    PyPlot.plot(xgrid, ygrid, label = "pdf(y)")
    PyPlot.vlines(median, 0, pdf(median), label = "median",  colors = "b")
    PyPlot.vlines(y0_true, 0, pdf(y0_true), label = "true value",  colors = "r")
    # plot CI
    if CI != nothing
        CI_id = (xgrid .> CI[1]) .* (xgrid .< CI[2])
        CI_xrange = vcat(CI1[1], xgrid[CI_id], CI1[2])
        CI_yrange = vcat(pdf(CI[1]), ygrid[CI_id], pdf(CI[2]))
        PyPlot.fill_between(CI_xrange, 0, CI_yrange, alpha = 0.3, label = "95% confidence interval")
    end
    PyPlot.legend(fontsize=8)
    PyPlot.grid()
    PyPlot.title(mytitle, fontsize=10)
#     PyPlot.savefig("Figure/Prediction.pdf")
end