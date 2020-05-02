include("sample.jl")
include("../statistics.jl")
"""
Generates initial guess for constrained optimization problem, i.e.
finds y value for which lb < constr(y;s) < ub

OUTPUT:
augmented [y, s] vector of value and 
"""
function init_constrained_pt(cdf_fixed, lx, ux; quantile = 0.25)
    (_, loc) = sample_points( _ -> 1.0, lx[2:end], ux[2:end]; num = 1) #samples random location in box
    #(_, loc) = sample( _ -> 1.0, lx[2:end], ux[2:end]; num = 1)
    cdf_y(y) = (arg = Base.vcat([y], reshape(loc, length(loc))); cdf_fixed(arg))
    #@info "loc" loc
    #@info "type of cdf_y", typeof(cdf_y)
    @info "loc", loc
    @info "lx", lx
    @info "ux", ux
    #try 
    (ystar, _, _) = findQuantile(cdf_y, [0, 6000]; p = quantile)
    #catch e
    #    println("findQuantile Error...")
    #    @info "cdf_y(2000)", cdf_y(2000)
    #    @info "cdf_y(0)", cdf_y(0)
    #finally
    println("successfully ran findQuantile...")
    vec = vcat(ystar, reshape(loc, length(loc)))
    @assert typeof(vec)<:Array{T, 1} where T
    return vec
    #end
end

function findQuantile(cdf::Function, support::Array{G,1} where G; pdf = nothing, pdf_deriv=nothing, p=.25) 
    bound = support
    #@info "bound", bound
    func = y0 -> cdf(y0) - p
    #@info "func0: ", func(0)
    #@info "func1: ", func(800)
    ystar = fzero(func, bound[1], bound[2]) 
    relerr = abs(p-cdf(ystar))/p
    @assert relerr < 1 e-1
    return ystar, relerr, bound
end






