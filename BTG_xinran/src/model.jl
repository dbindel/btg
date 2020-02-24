using LinearAlgebra
using Random
using Distributions
using Cubature
using Roots
using SpecialFunctions

include("kernels/kernel.jl")
include("kernels/mean_basis.jl")
include("transform/nonlinfun.jl")
include("quadrature/quadgrid.jl")
include("param/param_prior.jl")
include("computation/param_grid_comp.jl")
include("computation/post_prob_comp.jl")
include("computation/weight_comp.jl")
include("computation/distribution_comp.jl")

# Inputs:
    # x0: where to predict, testing data
    # training_data: including features/index/x and labels/values/z
    # kernel: a specific RBF kernel
    # nonlinfun: a specific nonlinear transformation, parameterized
    # p: the degree of freedom using in polynomial basis for the prior mean
    #    p = 1 by default (constant prior mean function)

# Outputs: 
    # the distribution of function values z0, at a given index x0
    # including mean, median, standard deviation and condidence interval


function model(x0, training_data, kernel, nonlinfun, p = 1)
    # training_data
    x = training_data[1]
    z = training_data[2]
    
    # define nonlinear transformation
    # both function values and derivatives
    g =  (x, lambda) -> nonlinfun(x, lambda)[1]
    dg = (x, lambda) -> nonlinfun(x, lambda)[2]

    # define polynomial basis
    X = meanbasisMat(x, p)
    X0 = meanbasisMat(x0, p)

    # precompute the constant coefficient in p(z0|z)
    n = size(x, 1) # #training points
    k = size(x0, 1) # #predicting points
    Gamma = gamma((n-p+k)/2)/(gamma((n-p)/2) * pi^(k/2))

    # wrap up basic training info
    traindata = (idx = x, val = z)
    nonlintrans = (fun = g, deriv = dg)
    trainBasicInfo = (data = training_data, kernel = kernel, 
                    polydof = p, nonlintrans = nonlintrans)

    ## compute weights
    # compute quadrature grids and weights
    n_quad = [8, 8, 10] # number of quad nodes for each param, could increase if needed
    n_theta = [n_quad[1], n_quad[2]]
    n_lambda = n_quad[3]
    n_param = (theta = n_theta, lambda = n_lambda)
    param_gridInfo = param_grid_comp(n_param, param_priorInfo)

    # compute alpha_ij = w_i w_j h_ij for theta_i and lambda_j
    alpha, sideSetInfo = weight_comp(param_gridInfo, param_priorInfo, trainBasicInfo)
  
    # compute distribution of z0 at x0
    distribution_z0 = distribution_comp(x0, trainBasicInfo, sideSetInfo, param_gridInfo, alpha, Gamma)
    
    return distribution_z0
end

