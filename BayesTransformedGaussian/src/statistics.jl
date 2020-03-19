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
  #= 
  Brief idea: bisection
    Suppose the target interval is [alpha*, beta*], i.e. integral of pdf on [alpha*, beta*] = wp
    and the corresponding height is pdf(alpha*) = pdf(beta*) = l_height*
    Say l* is the horizontal line intersecting with pdf curve at alpha* and beta*, 
    we first find two horizontal l lines sitting on each side of l*
    One is lower with height l_height_low, and one is higher with height l_height_high
    Then we use a bisection way to gradually find the target l*
  =#
     
  #= Notations:
  [alpha, beta]: 
      interval for integration, target credible interval
      s.t. 1) pdf(alpha) = pdf(beta) and 2) integral of pdf from alpha to beta = wp (like 0.95)
  l_height_low/l_height_high: 
      height of the l line that is lower/higher than the target one
  alpha_low/alpha_high, beta_low/beta_high: 
      the corresponding interval
  int_low/int_high: 
      the corresponding integral value, i.e. integral of pdf on [alpha_low/high, beta_low/high]
  hquadrature: 
      function from the package Cubature for numerical integration
      e.g. integral_heightue, error = hquadrature(f, a, b) gives integral of f on [a,b] with some error
      we mostly use this to compute integral of pdf on [alpha, beta] for different alpha and beta
  =#
   
  # assume we have the mode
  mode_d = mode(btg, x)
  l_height_low = 1e-3
  α_low, β_low, int_low = find_αβ(l_height_low, [0, mode_d], [mode_d, bound])
  # adjust height if l low is not lower than l* (i.e. int < wp)
  l_height_low, α_low, β_low, int_low = adjust(l_height_low, α_low, β_low, int_low, "low")

  l_height_high = f(mode_d) - 1e-6
  α_high, β_high, int_high = find_αβ(l_height_high, [0, mode_d], [mode_d, bound])
  # adjust height if l_high is not higher than l* (i.e. int > wp)
  l_height_high, α_high, β_high, int_high = adjust(l_height_high, α_high, β_high, int_high, "high")

  α_mid = 0.
  β_mid = 0.
  l_mid = 0.
  N = 0
  int = int_low
  while !isapprox(int, wp) && N < 50
  l_mid = (l_height_high + l_height_low)/2
  α_mid, β_mid, int = find_αβ(l_mid, [α_low, α_high], [β_high, β_low])
  if int > wp
      l_height_low = l_mid
      α_low = α_mid
      β_low = β_mid
  else
      l_height_high = l_mid
      α_high = α_mid
      β_high = β_mid
  end
  N += 1
  end
  return [α_mid, β_mid, l_mid]

  # helper functions
  #= 
  given a height l_height, find the two intersections s.t. pdf(α) = pdf(β) = l_height
  Input:
    l_height: given height
    α_intvl/β_intvl: interval for root finding for alpha/beta
  Output:
    [α, β]: the two intersections
    int: integral value of pdf on [α, β]
  =#
  function find_αβ(l_height, α_intvl, β_intvl)
    # find α and β within ginve intervals α_intvl and β_intvl respectively
    routine_α = optimize(x -> abs(f(x) - l_height), α_intvl[1], α_intvl[2], GoldenSection())
    α = Optim.minimizer(routine_α)
    routine_β = optimize(x -> abs(f(x) - l_height), β_intvl[1], β_intvl[2], GoldenSection())
    β = Optim.minimizer(routine_β)
    int = hquadrature(f, α, β)[1]
    return α, β, int
  end

  #= 
  Adjust the height in case initial choice of low/high is not proper 
    Since we want int_low > wp and int_high < wp to do bisection, 
    we have to adjust height if initial int_low < wp or int_high > wp
  Input:
    l: current height
    [α, β]: current interval
    int: current integral value
    MODE: indicates we are adjusting the low line or the high line
  Output:
    According values after adjustment
    s.t. int > wp if MODE == "low"; int < wp else.
 =#
  function adjust(l, α, β, int, MODE)
    # adjust if int < wp
    if MODE == "low"
        while int < wp
            l /= 2
            α, β, int = find_αβ(l, [0, α], [β, bound])
        end
    else
        while int > wp
            l = (l + f(mode_d))/2
            α, β, int = find_αβ(l, [α, mode_d], [mode_d, β])
        end
    end
    return l, α, β, int
  end
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
