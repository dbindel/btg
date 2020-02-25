# plot the distribution at one single point x0
using PyPlot
function plot_distribution_single(distribution_z0, x0, z0_true)
    pdf0 = distribution_z0.pdf
    mean = distribution_z0.mean
    # median = distribution_z0.median
    # CI = distribution_z0.CI
    # CI_left = CI[2]
    # CI_right = CI[3]
    
    # determine the range of plotting
    kmax = 20
    tol = 1e-3
    b0 = 5.0
    b, iter = zero_finding(pdf0, kmax, tol, b0)

    # plot pdf
    z_grid = range(1e-5, stop = b, step = 0.01)
    p_grid = pdf0.(z_grid);
    PyPlot.plot(z_grid, p_grid, label = "probability density function")
    # plot mean median and true value
    PyPlot.vlines(mean, 0, pdf0(mean), label = "mean", colors = "k")
    # PyPlot.vlines(median, 0, pdf0(median), label = "median",  colors = "b")
    PyPlot.vlines(z0_true, 0, pdf0(z0_true), label = "true value",  colors = "r")
    # plot CI
    # CI_x_range = range(CI_left, stop = CI_right, step = 0.01)
    # CI_y_range = pdf0.(CI_x_range)
    # PyPlot.fill_between(CI_x_range, 0, CI_y_range, alpha = 0.3, label = "95% confidence interval")
    PyPlot.legend()
    PyPlot.grid()
    PyPlot.title("Predicted Distribution at x = $x0")
#     PyPlot.savefig("Figure/Prediction")
end