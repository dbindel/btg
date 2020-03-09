
using LinearAlgebra
using Random
using Distributions
using Roots
using Cubature
    
include("../computation/post_prob_comp.jl")
include("../root_finding/my_bisection.jl")
include("../root_finding/zero_finding.jl")

# single point prediction
# given x_pred, compute the pdf of Z at each index
# also compute median and 95% CI and each index
function distribution_comp(x_pred, trainBasicInfo, sideSetInfo, param_gridInfo, alpha, Gamma)
    m = size(x_pred, 1)
    @assert m == 1
    dim = size(x_pred, 2)

    # compute the pdf
    pdf = z0 -> prob = pdf_z0(x_pred, z0, trainBasicInfo, sideSetInfo, param_gridInfo, alpha, Gamma)
    # find the upper bound for the pdf integral interval
    b0 = 30.0 # initial guess of the interval [0, b0] for the pdf integral
    kmax = 20
    tol = 1e-9
    b, iter = zero_finding(pdf, kmax, tol, b0)

    # compute mean, median, standard deviation, etc.
    # mean_z, median_z, stdv_z = median_comp(pdf, b)
    mean_z, stdv_z = median_comp(pdf, b)

    # compute confidence interval
    # p = 0.95 # by default we compute the 95% confidence interval
    # q = [0.8, 1.2]
    # eps = 1e-6
    # z2_initl = median_z + stdv_z
    # l_initl = pdf(z2_initl)
    # z1_initl = my_bisection(x -> pdf(x) - l_initl, 1e-3, median_z)
    # # z1_initl = find_zero(x -> pdf_temp(x) - l_initl, median_z - stdv_z)
    # initl = [l_initl, z1_initl, z2_initl]
    # CI = CI_comp_BTG(initl, p, q, pdf, eps, zu_new, median_z)

    # currently, median, mode and CI are not included
    dist_z = (mean = mean_z, stdv =  stdv_z, pdf = pdf, upperbd = b)
    return dist_z
end



function median_comp(pdf_z, b)
    # compute the mean 
    mean_z = hquadrature(x -> x * pdf_z(x), 1e-5, b; reltol=1e-8, abstol=0, maxevals=0)[1]

    # compute the median 
    # medianfun = x ->  hquadrature(pdf_z, 1e-5, x; reltol=1e-8, abstol=0, maxevals=0)[1] - 0.5
    # median_z = find_zero(medianfun, mean_z)

    # standard deviation
    variancefun = x -> (x - mean_z)^2 * pdf_z(x)
    variance_z = hquadrature(variancefun, 1e-5, b; reltol=1e-8, abstol=0, maxevals=0)[1] 
    stdv_z = sqrt(variance_z)

    # return mean_z, median_z, stdv_z
    return mean_z, stdv_z

end

# compute confidence interval
function CI_comp_BTG(initl, p, q, pdf_z, eps, zu_new, median_z)
    l = initl[1]
    y1 = initl[2]
    y2 = initl[3]
    int = hquadrature(x -> pdf_z(x), y1, y2; reltol=1e-8, abstol=0, maxevals=0)
    Int = int[1]
    println("initial Int = $Int")
    flush(stdout)
    if int[1] < p
       while int[1] < p
            l_new = q[1] * l
            # println("1, l_new = $l_new")
            # flush(stdout1)
            # y1_new = find_zero(x -> pdf_z(x) - l_new, y1)
            # y2_new = find_zero(x -> pdf_z(x) - l_new, y2)
            y1_new = my_bisection(x -> pdf_z(x) - l_new, 1e-3, y1)
            y2_new = my_bisection(x -> pdf_z(x) - l_new, y2, zu_new)
            int = hquadrature(x -> pdf_z(x), y1_new, y2_new;
                                reltol=1e-8, abstol=0, maxevals=0)
            l = l_new
            y1 = y1_new
            y2 = y2_new
            CurentInt = int[1]
            println("Current Int = $CurentInt")
            flush(stdout)
        end
        l_low = l
        y1_low = y1
        y2_low = y2
        l_up = initl[1]
        y1_up = initl[2]
        y2_up = initl[3]
    elseif int[1] > p
        while  int[1] > p
            l_new = q[2] * l
            # println("2, l_new = $l_new")
            # flush(stdout1)
            # y1_new = find_zero(x -> pdf_z(x) - l_new, y1)
            # y2_new = find_zero(x -> pdf_z(x) - l_new, y2)
            y1_new = my_bisection(x -> pdf_z(x) - l_new, y1, median_z)
            y2_new = my_bisection(x -> pdf_z(x) - l_new, median_z, y2)
            int = hquadrature(x -> pdf_z(x), y1_new, y2_new;
                                reltol=1e-8, abstol=0, maxevals=0)
            l = l_new
            y1 = y1_new
            y2 = y2_new
        end
        l_low = initl[1]
        y1_low = initl[2]
        y2_low = initl[3]
        l_up = l
        y1_up = y1
        y2_up = y2
    end
    println("l_low = $l_low, y1_low = $y1_low, y2_low = $y2_low")
    flush(stdout)
    println("l_l_uplow = $l_up, y1_up = $y1_up, y2_up = $y2_up")
    flush(stdout)


    #  initialize middle points
    l_mid = 0
    y1_mid = 0
    y2_mid = 0
    N = 0
    # now with lower and upper bound, do sth like bisection... 
    while (abs(int[1] - p) > eps) && N < 100
        l_mid = (l_up + l_low)/2
        # y1_mid = find_zero(x -> pdf_z(x) - l_mid, (y1_up + y1_low)/2)
        # y2_mid = find_zero(x -> pdf_z(x) - l_mid, (y2_up + y2_low)/2)
        y1_mid = my_bisection(x -> pdf_z(x) - l_mid, y1_low, y1_up)
        y2_mid = my_bisection(x -> pdf_z(x) - l_mid, y2_up, y2_low)
        int = hquadrature(x -> pdf_z(x), y1_mid, y2_mid; reltol=1e-8, abstol=0, maxevals=0)
        if int[1] > p
            l_low = l_mid
            y1_low = y1_mid
            y2_low = y2_mid
        else
            l_up = l_mid
            y1_up = y1_mid
            y2_up = y2_mid
        end
        N += 1
    end
    CI = [l_mid, y1_mid, y2_mid]
    return CI
end
