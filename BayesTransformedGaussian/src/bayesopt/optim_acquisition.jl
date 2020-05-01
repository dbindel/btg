include("../covariatefun.jl")
using Optim

"""
cdf(u, s) is a function of location s and value u, gradients and hessian are
w.r.t the vector (u, s).

INPUTS:
cdf - takes in (x, Fx, y) and outputs cdf
cdf_gradient - takes in (x, Fx, y) and outputs Jacobian
cdf_hessian - takes in (x, Fx, y) and outputs Hessian

OUTPUTS:

"""
function optimizeUCB(cdf, cdf_gradient, cdf_hessian, lx, ux)
    #v is augmented vector consisting of [u, s], where u is value and s is location

    cdf_fixed(v) = cdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
    cdf_gradient_fixed(v) = cdf_gradient(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
    cdf_hessian_fixed(v) = cdf_hessian(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
    d = length(lx)  #location is d-dimensional, while label y is 1-dimensional, so augmented vector is (d+1)-dimensional
    fun(x) = x[1] #x\in R^d+1  
    function fun_grad!(g, x)
        g[1] = 1
        for i = 2:d
            g[i] = 0 
        end
    end;
    function fun_hess!(h, x)
        for i = 1:d
            for j = 1:d
                h[i, j] = 0
            end
        end
    end;
    con_c!(c, x) = (c[1] = cdf_fixed(x); c)
    function con_jacobian!(J::Array{T} where T <:Float64, x)
        grad = cdf_gradient_fixed(x)
        for i = 1:d
            J[i] = grad[i]
        end
        J
    end;
    function con_h!(h, x, λ)
        hess = cdf_hessian_fixed(x)
        for i = 1:d
            for j = 1:d
                h[i, j] += λ[1]*hess[i, j]
            end
        end
    end;
    #lx = [-5.0, -5.0]; ux = [5.0, 5.0] #function box constraint
    lc = [0.2]; uc = [0.3] #constraint on cdf by default, since we want 1 stdev 
    x0 = init_constrained_pt(cdf_fixed, lx[2:end], ux[2:end]; quantile = 0.25)
    @info "x0 found with init_constr_point", x0
    @info "cdf_fixed(x0)", cdf_fixed(x0)
    df = TwiceDifferentiable(fun, fun_grad!, fun_hess!, x0)
    @info "lx" lx
    @info "ux" ux
    @info "lc" lc
    @info "uc" uc
    dfc = TwiceDifferentiableConstraints(con_c!, con_jacobian!, con_h!, lx, ux, lc, uc)
    @info x0
    res = optimize(df, dfc, x0, IPNewton())
    return (res.minimizer, res.minimum)
end





