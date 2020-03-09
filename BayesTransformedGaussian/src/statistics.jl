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
  # Step 1: find an initial interval [alpha0, beta0] to start
  median = quantile(btg, x, p=0.5)
  l_fun(x) = pdf(btg, x)
  beta = 1.1 * median
  l_val = l_fun(beta)
  l_fun_equal(x) = abs(l_fun(x) - l_val)
  # find alpha such that p(alpha) = p(beta)
  routine = optimize(l_fun_equal, 1e-10, median) 
  alpha = Optim.minimizer(routine)
  int = hquadrature(l_fun, alpha, beta)[1]

  # Step 2: find another interval [alpha1, beta1] 
  # s.t. (p(alpha1) - wp)(p(alpha0) - wp) < 0
  if int < wp
    while int < wp
        l_up = l_val
        alpha_up = alpha
        beta_up = beta
        int_up = int
        
        l_val *= 0.6
        routine_alpha = optimize(x -> abs(l_fun(x) - l_val), 1e-10, alpha) 
        alpha = Optim.minimizer(routine_alpha)
        routine_beta = optimize(x -> abs(l_fun(x) - l_val), beta, b)
        beta = Optim.minimizer(routine_beta)
        int = hquadrature(l_fun, alpha, beta)[1]
        println("Current tuple [$alpha, $beta, $l_val, $int]")
    end
    l_low = l_val
    alpha_low = alpha
    beta_low = beta
    int_low = int
  else
    while int > wp
      l_low = l_val
      alpha_low = alpha
      beta_low = beta
      int_low = int
      
      l_val *= 2
      routine_alpha = optimize(x -> abs(l_fun(x) - l_val), alpha, mean_temp) 
      alpha = Optim.minimizer(routine_alpha)
      routine_beta = optimize(x -> abs(l_fun(x) - l_val), mean_temp, beta)
      beta = Optim.minimizer(routine_beta)
      int = hquadrature(l_fun, alpha, beta)[1]
      l_low = l_val
      alpha_low = alpha
      beta_low = beta
      int_low = int
      println("Current tuple \n [$alpha, $beta, $l_val, $int]")
    end
    l_up = l_val
    alpha_up = alpha
    beta_up = beta
    int_up = int
  end

  # Step 3: find middle point [alpha, beta] 
  # s.t. cdf(beta) - cdf(alpha) = wp 
  l_mid = 0.
  alpha_mid = 0.
  beta_mid = 0.
  N = 0

  while !isapprox(int, wp) && N < 100
    l_mid = (l_up + l_low)/2
    alpha_mid = Optim.minimizer(optimize(x -> abs(l_fun(x) - l_mid), alpha_low, alpha_up))
    beta_mid = Optim.minimizer(optimize(x -> abs(l_fun(x) - l_mid), beta_up, beta_low))
    int = hquadrature(l_fun, alpha_mid, beta_mid)[1]
    if int > wp
        l_low = l_mid
        alpha_low = alpha_mid
        beta_low = beta_mid
    else
        l_up = l_mid
        alpha_up = alpha_mid
        beta_up = beta_mid
    end
    N += 1
  end
  @assert isapprox(l_fun(alpha_mid), l_fun(beta_mid)) "Should have equal value of pdf function."
  @assert isapprox(int, wp) "Should have $wp confidence interval"
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
