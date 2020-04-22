using Optim
using Cubature
using Roots
using StatsFuns

# import Statistics: median, pdf, cdf
"pre-process pdf and cdf, given fixed pdf and cdf at x0, compute estimated support and check if pdf is proper"
function pre_process(x0::Array{T,2}, Fx0::Array{T,2}, pdf::Function, cdf::Function, dpdf::Function, quantbound::Function) where T<:Float64
    quantbound_fixed(p) = quantbound(x0, Fx0, p) 
    pdf_fixed(y) = pdf(x0, Fx0, y)
    cdf_fixed(y) = cdf(x0, Fx0, y)
    dpdf_fixed(y) = dpdf(x0, Fx0, y)
    support = [.1, 5.]
    function support_comp!(pdf, support)
      current = pdf(support[1])
      for i in 1:14
        next = pdf(support[1]/10)
        if next < current
          support[1] /= 10
        else
          break
        end
        current = next
      end
      while pdf(support[2]) > 1e-6 
        support[2] *= 1.2
      end
      # should make sure CDF(support[2]) - CDF(support[1]) > .96 to make 95% CI possible
      INT = cdf_fixed(support[2]) - cdf_fixed(support[1])
      @assert INT > .95 "pdf integral $INT"
      return nothing
    end
    support_comp!(pdf_fixed, support)
    return (pdf_fixed, cdf_fixed, dpdf_fixed, quantbound_fixed, support)
end

"wrap up all statistics computation"
function summary_comp(pdf_fixed::Function, cdf_fixed::Function, dpdf_fixed::Function, quantbound_fixed::Function, support::Array{T,1};
                       px = .5, confidence_level = .95) where T<:Float64
    quant_p, error_quant = quantile(cdf_fixed, quantbound, support; p=px)
    med, error_med = median(cdf_fixed, quantbound, support)
    mod = mode(pdf_fixed, support)
    CI_equal, error_CI_eq = credible_interval(cdf_fixed, quantbound, support; 
                                                mode=:equal, wp = confidence_level)
    # CI_narrow, error_CI_nr = credible_interval(cdf_fixed, quantbound, support;
    #                                             mode=:narrow, wp = confidence_level)
    quantileInfo = (level = px, value = quant_p, error = error_quant)
    medianInfo = (value = med, error = error_med)
    CIequalInfo = (equal = CI_equal, error = error_CI_eq)
    # CInarrowInfo = (equal = CI_narrow, error = error_CI_nr)
    CInarrowInfo = nothing
    DistributionInfo = (quantile = quantileInfo, median = medianInfo, mode = mod, CIequal = CIequalInfo, CInarrow = CInarrowInfo)
    return DistributionInfo
end


"""
Given pdf, cdf and maybe pdf_deriv, 
compute median, quantile, mode, symmetric/narrowest credible interval.
Warning: only for normalized values
"""
function median(cdf::Function, quantbound::Function, support::Array{T,1}; pdf = nothing, pdf_deriv=nothing) where T<:Float64
    med, err = quantile(cdf, quantbound, support)
    return med, err, bound
end

function quantile(cdf::Function, quantbound::Function, support::Array{T,1}; pdf = nothing, pdf_deriv=nothing, p::T=.5) where T<:Float64
    bound = support
    try 
      bound = quantbound(p)
    catch err
    end
    quant = fzero(y0 -> cdf(y0) - p, bound[1], bound[2]) 
    err = abs(p-cdf(quant))/p
    # status = err < 1e-5 ? 1 : 0
    return quant, err, bound
end

function mode(pdf::Function, support::Array{T,1}; cdf = nothing, pdf_deriv=nothing) where T<:Float64
    # maximize the pdf 
    routine = optimize(x -> -pdf(x), support[1], support[2]) 
    mod = Optim.minimizer(routine)
    return mod
end

function credible_interval(cdf::Function, quantbound::Function, support::Array{T,1}; 
                            pdf=nothing, pdf_deriv=nothing, wp::T=.95, mode=:equal) where T<:Float64
    return credible_interval(cdf, quantbound, support, Val(mode); pdf_deriv=pdf_deriv, wp=wp)
end

function credible_interval(cdf::Function, quantbound::Function, support::Array{T,1}, ::Val{:equal};  
                            pdf=nothing, pdf_deriv=nothing, wp::T=.95) where T<:Float64
    lower_qp = (1 - wp) / 2
    upper_qp = 1 - lower_qp
    lower_quant = quantile(cdf, quantbound, support; p=lower_qp)[1]
    upper_quant = quantile(cdf, quantbound, support; p=upper_qp)[1]
    err = abs(cdf(upper_quant) -  cdf(lower_quant) - wp)/wp
    return [lower_quant, upper_quant], err
end

function credible_interval(cdf::Function, quantbound::Function, support::Array{T,1}, ::Val{:narrow}; 
                            pdf=nothing, pdf_deriv=nothing, wp::T=.95) where T<:Float64
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
  bound:
      the bound of support of the pdf, currently assume this is available
      i.e. pdf(x) = 0 if z > bound
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
    temp_fun(x) = pdf(x) - l_height
    α = fzero(temp_fun,  α_intvl[1], α_intvl[2])
    β = fzero(temp_fun,  β_intvl[1], β_intvl[2])
    int = hquadrature(x -> pdf(x), α, β)[1]
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
            α, β, int = find_αβ(l, [support[1], α], [β, support[2]])
        end
    else
        while int > wp
            l = (l + pdf(mode_d))/2
            α, β, int = find_αβ(l, [α, mode_d], [mode_d, β])
        end
    end
    return l, α, β, int
  end
  mode_d = mode(pdf, cdf, support)
  l_height_low = pdf(support[1]) 
  α_low = support[1]
  β_low = fzero(x -> pdf(x) - l_height_low,  mode_d, support[2])
  int_low = hquadrature(x -> pdf(x), α_low, β_low)[1]
  # α_low, β_low, int_low = find_αβ(l_height_low, [support[1], mode_d], [mode_d, support[2]])
  # adjust height if l low is not lower than l* (i.e. int < wp)
  # l_height_low, α_low, β_low, int_low = adjust(l_height_low, α_low, β_low, int_low, "low")
  l_height_high = pdf(mode_d)*0.9
  α_high, β_high, int_high = find_αβ(l_height_high, [α_low, mode_d], [mode_d, β_low])
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
  err1 = abs(cdf(β_mid) -  cdf(α_mid) - wp)/wp
  err = max(err1, abs(pdf(α_mid)- pdf(β_mid))/abs(pdf(β_mid)))
  return ([α_mid, β_mid], err)

 
end
    

# @doc raw"""
# """
# function map_estimate(btg::BTG)
#     # TODO
# end

# @doc raw"""
# """
# function cross_validate(btg::BTG)
#     # TODO
# end
