
using LinearAlgebra
using Random
using Distributions
using Roots
using Cubature
    
include("../computation/post_prob_comp.jl")
include("../root_finding/my_bisection.jl")

# given a set of indices x_pred, compute the pdf of Z at each index
# also compute median and 95% CI and each index
function distribution_comp(x_pred, trainBasicInfo, sideSetInfo, param_gridInfo, alpha, Gamma)
m = size(x_pred, 1)
dim = size(x_pred, 2)
# not sure how to estimate the right bound of Z
zu = 30

# by default we compute the 95% confidence interval
p = 0.95
q = [0.8, 1.2]
eps = 1e-6

# if predicting one single point 
if  m == 1 
    pdf = z0 -> prob = pdf_z0(x_pred, z0, trainBasicInfo, sideSetInfo, param_gridInfo, alpha, Gamma)
    # compute mean median and standard deviation
    mean_z, median_z, stdv_z, zu_new = median_comp(pdf, zu)
    # println("mean = $mean_z")
    # println("median = $median_z")
    # println("stdv = $stdv_z")
    # println("zu_new = $zu_new")

    ## check if integral of pdf is 1
    # INT = hquadrature(pdf, 1e-5, zu_new)[1]
    # println("Total integal is $INT")
    # flush(stdout)
################################################################
    # compute confidence interval
    # z2_initl = median_z + stdv_z
    # l_initl = pdf(z2_initl)
    # z1_initl = my_bisection(x -> pdf(x) - l_initl, 1e-3, median_z)
    # # z1_initl = find_zero(x -> pdf_temp(x) - l_initl, median_z - stdv_z)
    # initl = [l_initl, z1_initl, z2_initl]
    # CI = CI_comp_BTG(initl, p, q, pdf, eps, zu_new, median_z)

    # # warp up
    # dist_z = (mean = mean_z, median = median_z, 
    #           stdv =  stdv_z, CI = CI, pdf = pdf)

################################################################
    # do not compute confidence interval
    # warp up
    dist_z = (mean = mean_z, median = median_z, 
              stdv =  stdv_z, pdf = pdf)

else 
    mean_set = zeros(m)
    median_set = zeros(m)
    stdv_set = zeros(m)
    CI_set = zeros(m, 3)   
    for i in 1:m    
        x_temp = x_pred[i, :]
        # println("At x = $x_temp, i = $i")
        pdf_temp = z0 -> prob = pdf_z0(x_temp, z0, trainBasicInfo, sideSetInfo, param_gridInfo, alpha, Gamma)
        mean_z, median_z, stdv_z, zu_new = median_comp(pdf_temp, zu)
        mean_set[i] = mean_z
        median_set[i] = median_z
        stdv_set[i] = stdv_z
        # println("mean = $mean_z")
        # println("mean = $median_z")
        # println("median = $stdv_z")

################################################################
        # compute confidence interval
        # z2_initl = median_z + stdv_z
        # l_initl = pdf_temp(z2_initl)
        # z1_initl = my_bisection(x -> pdf_temp(x) - l_initl, 1e-3, median_z)
        # initl = [l_initl, z1_initl, z2_initl]
        # CI = CI_comp_BTG(initl, p, q, pdf_temp, eps, zu_new, median_z)
        # CI_set[i, :] = CI

      
    end
    # compute CI
    # dist_z = (mean = mean_set, median = median_set, 
    #           stdv =  stdv_set, CI = CI_set)

    # do not compute CI
    dist_z = (mean = mean_set, median = median_set, 
              stdv =  stdv_set)
end

    return dist_z

end



function median_comp(pdf_z, xu)
    # make sure the integral of pdf is near 1
    k = 1
    int = hquadrature(pdf_z, 1e-5, k*xu; reltol=1e-8, abstol=0, maxevals=0)
    # println("At first, integral of pdf_z is $int, k = $k")
    while (abs(1-int[1]) > 0.02 && k < 20)
        k = k+0.5
        int = hquadrature(pdf_z, 1e-5, k*xu; reltol=1e-8, abstol=0, maxevals=0)
    end
    # println("Integral of pdf_z is $int, k = $k")

    # compute the mean 
    mean_z = hquadrature(x -> x * pdf_z(x), 1e-5, k*xu; reltol=1e-8, abstol=0, maxevals=0)[1]

    # compute the median 
    medianfun = x ->  hquadrature(pdf_z, 1e-5, x; reltol=1e-8, abstol=0, maxevals=0)[1] - 0.5
    median_z = find_zero(medianfun, mean_z)

    # standard deviation
    variancefun = x -> (x - mean_z)^2 * pdf_z(x)
    variance_z = hquadrature(variancefun, 1e-5, k*xu; reltol=1e-8, abstol=0, maxevals=0)[1] 
    stdv_z = sqrt(variance_z)

    return mean_z, median_z, stdv_z, k*xu

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
