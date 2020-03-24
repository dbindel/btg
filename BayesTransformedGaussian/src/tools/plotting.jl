using PyPlot
using Plots

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
    display(Plots.plot(x, y, label = label))
end

function plt!(f, a, b, numpts=100, label = "y")
    h = (b-a)/(numpts-1)
    y = zeros(1,numpts)
    x = collect(a:h:b)
    y = map(f, x)
    display(Plots.plot!(x, y, label = label))
end

"""
plot pdf, median, mode and credible interval
"""
function plot_distribution_single(pdf, median, mode, CI, z0_true, quadtype)
    # plot pdf
    b = ceil(2.5 * mode)
    z_grid = range(1e-5, stop = b, step = 0.01)
    p_grid = pdf.(z_grid);
    PyPlot.plot(z_grid, p_grid, label = "probability density function")
    # PyPlot.vlines(median, 0, pdf0(mean), label = "mean", colors = "k")
    PyPlot.vlines(median, 0, pdf(median), label = "median",  colors = "b")
    PyPlot.vlines(mode, 0, pdf(mode), label = "mode")
    PyPlot.vlines(z0_true, 0, pdf(z0_true), label = "true value",  colors = "r")
    # plot CI
    CI_x_range = range(CI[1], stop = CI[2], step = 0.01)
    CI_y_range = pdf.(CI_x_range)
    PyPlot.fill_between(CI_x_range, 0, CI_y_range, alpha = 0.3, label = "95% confidence interval")
    PyPlot.legend(fontsize=8)
    PyPlot.grid()
    PyPlot.title(quadtype, fontsize=10)
#     PyPlot.savefig("Figure/Prediction")
end

