include("sample.jl")
include("../statistics.jl")
"""
Generates initial guess for constrained optimization problem, i.e.
finds y value for which lb < constr(y;s) < ub
"""
function init_constrained_pt(cdf_fixed, lx, ux; quantile = 0.25)
    (_, loc) = sample( _ -> 1.0, lx, ux; num = 1) #samples random location in box
    cdf_y(y) = (@info "y", y ; arg = Base.vcat([y], reshape(loc, length(loc))); @info "arg", arg; cdf_fixed(arg))
    @info "loc" loc
    @info "type of cdf_y", typeof(cdf_y)
    (ystar, _, _) = findQuantile(cdf_y, [0, 800]; p = 0.25)
    vec = vcat(ystar, reshape(loc, length(loc)))
    @assert typeof(vec)<:Array{T, 1} where T
    return vec
end

function findQuantile(cdf::Function, support::Array{G,1} where G; pdf = nothing, pdf_deriv=nothing, p=.25) 
    bound = support
    @info "bound", bound
    func = y0 -> cdf(y0) - p
    @info "func0: ", func(0)
    @info "func1: ", func(1)
    ystar = fzero(func, bound[1], bound[2]) 
    relerr = abs(p-cdf(ystar))/p
    @assert relerr < 1 e-1
    return ystar, relerr, bound
end



