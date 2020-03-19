using Optim
using Cubature

import Statistics: median, pdf, cdf

@doc raw"""
  Add documentation
"""
function pdf(btg::BTG, x::AbstractVector{R}) where R <: Real
    pdf_point_eval = compute_dists!(btg, x)[1]
    return pdf_point_eval
end

@doc raw"""
  Add documentation
"""
function cdf(btg::BTG, x::AbstractVector{R}) where R <: Real
    cdf_point_eval = compute_dists!(btg, x)[2]
    return cdf_point_eval
end

@doc raw"""
  Add documentation
"""
function median(btg::BTG, x::AbstractVector{R}) where R <: Real
    med = quantile(btg, x)
    return med
end


@doc raw"""
  Add documentation
"""
function quantile(btg::BTG, x::AbstractVector{R}; p::R=.5) where R <: Real
    cdf(x) = cdf(btg, x)
    quantile_func(x) = cdf(x) - p

    function quantile_deriv!(storage, x)
        storage[1] = pdf(btg, x)
    end

    initial_guess = [ rand() ]
    routine = optimize(quantile_func, quantile_deriv!, initial_guess, BFGS())

    quant = Optim.minimizer(routine)

    return quant
end

@doc raw"""
  Add documentation
"""
function mode(btg::BTG, x::AbstractVector{R}) where R <: Real
    # TODO
end

@doc raw"""
"""
function credible_interval(btg::BTG, x::AbstractVector{R}, wp::R; mode=:narrow) where R <: Real
    return credible_interval(btg, x, wp, Val(mode))
end

function credible_interval(btg::BTG, x::AbstractVector{R}, wp::R, ::Val{:equal}) where R <: Real
    lower_qp = (1 - wp) / 2
    upper_qp = 1 - lower_qp
    lower_quant = quantile(btg, x, p=lower_qp)
    upper_quant = quantile(btg, x, p=upper_qp)
    return [lower_quant, upper_quant]
end

function credible_interval(btg::BTG, x::AbstractVector{R}, wp::R, ::Val{:narrow}) where R <: Real
  #= Notations:
  [alpha, beta]: 
      interval for integration, target credible interval
      s.t. 1) pdf(alpha) = pdf(beta) and 2) integral of pdf from alpha to beta = wp (like 0.95)
  l_height: 
    value of pdf at some point, height of current l line
  l_fun_equal = abs(l(x) - l_height): 
      minimize this absolute value is to find another point s.t. l(x) = l_height 
      i.e. given a height of the l line, find an intersection of the horizontal l line and pdf curve
  hquadrature: 
      function from the package Cubature for numerical integration
      e.g. integral_heightue, error = hquadrature(f, a, b) gives integral of f on [a,b] with some error
      we mostly use this to compute integral of pdf on [alpha, beta] for different alpha and beta
  l_height_low/l_height_high: 
      height of an l line that is lower/higher than the target one
  alpha_low/alpha_high, beta_low/beta_high: 
      the corresponding interval

  =#

  #= 
  Brief idea: bisection
    Suppose the target interval is [alpha*, beta*], i.e. integral of pdf on [alpha*, beta*] = wp
    and the corresponding height is pdf(alpha*) = pdf(beta*) = l_height*
    We call the horizontal line passing through pdf(alpha*) and pdf(beta*) the target l line, l*
    We first find two horizontal l lines sitting on each side of l*
    and they are called l_height_low and l_high
    Then we use a bisection way to gradually find the target l*
  =#

  #= 
  Step 1: 
    find an initial interval [alpha, beta] to start
    such that pdf(alpha) = pdf(beta), i.e. same height
    also compute the integral of pdf on current [alpha, beta] 
  =#
  median = quantile(btg, x, p=0.5)
  # set initial beta to be a little larger than median, no specific reason, just a start
  beta = 1.1 * median
  l_height =  pdf(btg, beta)
  l_fun_equal(x) = abs(pdf(btg, x) - l_height)
  # find alpha such that pdf(alpha) = pdf(beta)
  routine = optimize(l_fun_equal, 1e-10, median) 
  alpha = Optim.minimizer(routine)
  # compute the integral of pdf on [alpha,beta]
  int = hquadrature(x -> pdf(btg, x), alpha, beta)[1]

  #= 
  Step 2: 
    find another interval [alpha, beta] 
    s.t. this l line and the initial l line found in step 1 are on different side of the target l line
    where the target l line means the credible interval we want
  =#

  if int < wp
    # int < wp means the interval is narrower than [alpha*, beta*] 
    # so it means that the line is higher than the target line l*
    while int < wp
        l_height_up = l_height
        alpha_up = alpha
        beta_up = beta
        int_up = int
        
        # if the current l line is higher than the target, 
        # we lower it until its lower than the target line
        l_height *= 0.6
        routine_alpha = optimize(l_fun_equal, 1e-10, alpha) 
        alpha = Optim.minimizer(routine_alpha)
        routine_beta = optimize(l_fun_equal, beta, b)
        beta = Optim.minimizer(routine_beta)
        int = hquadrature(x -> pdf(btg, x), alpha, beta)[1]
        println("Current tuple [$alpha, $beta, $l_height, $int]")
    end
    l_height_low = l_height
    alpha_low = alpha
    beta_low = beta
    int_low = int
  else
    # int > wp means the interval is larger than [alpha*, beta*] 
    # so it means that the line is lower than the target line l*
    while int > wp
      l_height_low = l_height
      alpha_low = alpha
      beta_low = beta
      int_low = int
      
      # if the current l line is higher than the target, 
      # we lower it until its lower than the target line
      l_height *= 2
      routine_alpha = optimize(l_fun_equal, alpha, mean_temp) 
      alpha = Optim.minimizer(routine_alpha)
      routine_beta = optimize(l_fun_equal, mean_temp, beta)
      beta = Optim.minimizer(routine_beta)
      int = hquadrature(x -> pdf(btg, x), alpha, beta)[1]
      l_height_low = l_height
      alpha_low = alpha
      beta_low = beta
      int_low = int
    end
    l_height_up = l_height
    alpha_up = alpha
    beta_up = beta
    int_up = int
  end


  # Step 3: Bisection to find a middle l line, intersecting with pdf at alpha and beta 
  # s.t. integral of pdf on [alpha, beta] = wp 
  l_mid = 0.
  alpha_mid = 0.
  beta_mid = 0.
  N = 0

  while !isapprox(int, wp) && N < 100
    l_mid = (l_height_up + l_height_low)/2
    alpha_mid = Optim.minimizer(optimize(x -> abs(l_fun(x) - l_mid), alpha_low, alpha_up))
    beta_mid = Optim.minimizer(optimize(x -> abs(l_fun(x) - l_mid), beta_up, beta_low))
    int = hquadrature(l_fun, alpha_mid, beta_mid)[1]
    if int > wp
        l_height_low = l_mid
        alpha_low = alpha_mid
        beta_low = beta_mid
    else
        l_height_up = l_mid
        alpha_up = alpha_mid
        beta_up = beta_mid
    end
    N += 1
  end

  # assert the constraint pdf(alpha) = pdf(beta)
  @assert isapprox(l_fun(alpha_mid), l_fun(beta_mid)) "Should have equal value of pdf function."
  # assert integral of pdf on [alpha, beta] = wp
  @assert isapprox(int, wp) "Should have $wp confidence interval"

  # return an array storing height of l and the credible interval [alpha_mid, beta_mid] found by bisection
  CI = [l_mid, alpha_mid, beta_mid]  

end
    
end

@doc raw"""
"""
function map_estimate(btg::BTG)
    # TODO
end

@doc raw"""
"""
function cross_validate(btg::BTG)
    # TODO
end
