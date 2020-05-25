function optimize_acqusition(cdf, cdf_gradient, lx, ux;  initial, maxiter = 300, quant = 0.25)
    #lx = [1, -5, -5]; ux = [3000.0, 5, 5] #box-constraints for optimization problem
    #make input an augmented vector [y, s]: label, location
    cdf_fixed(v) = cdf(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
    cdf_gradient_fixed(v) = cdf_gradient(v[2:end], linear_polynomial_basis(v[2:end]), v[1])
    d = length(lx)
    fun(x...) = x[1] #x\in R^d+1 
    dfun(g, x...) = (g[1] = 1; g[i] = 0 for i = 2:length(g);) 
    #cdf_wrapper(x) = cdf_fixed(collect(x)) 
    cdf_wrapper(x...) = cdf_fixed(collect(x))
    #cdf_gradient_wrapper(g, x...) = (grad = cdf_gradient_fixed(collect(x...)); g[i] = grad[i] for i = 1:length(grad); @info grad; @info g; return nothing)
    function cdf_gradient_wrapper(g, x...)
        grad = cdf_gradient_fixed(collect(x))
        @info "grad", grad
        for i = 1:length(grad)
            #@info "i", i
            #@info "grad component i", grad[i]
            global g[i] = grad[i]
        end
        return nothing
    end
    model = Model(Ipopt.Optimizer)
    set_optimizer_attributes(model, "tol" => 1e-2, "max_iter" => maxiter)
    #model = Model(Gurobi.Optimizer)
    #NLoptSolver(algorithm=:SLSQP))
    register(model, :fun, 3, fun, dfun)
    register(model, :cdf_wrapper, 3, cdf_wrapper, cdf_gradient_wrapper)

    @variable(model, lx[i] <= au[i=1:3] <= ux[i])
    #@variable(model, au)
    @NLobjective(model, Max, fun(au...))
    @NLconstraint(model, cdf_wrapper(au...) == quant)

    function single_optimization()
        initval = init_constrained_pt(cdf_fixed, lx, ux; initial = initial, quantile = quant)
        set_start_value(au[1], initval[1])
        set_start_value(au[2], initval[2])
        set_start_value(au[3], initval[3])
        @info "init_val", initval
        @info "cdf_fixed(init_val)" cdf_fixed(initval)
        #println("Running Optimizer...")
        #initval = [200, 4, 4]              
        #@info "initval", initval
        #@variable(model, lx[i] <= au[i=1:3] <= ux[i])
        JuMP.optimize!(model)
        println(value.(au))
        vstar = value.(au)
        return (vstar, initval)
    end
    NUM_RESTARTS = 10
    count = 1
    vstar = undef
    initval = undef
    while count <=10 #max 
        (vstar_cur, cur_initval) = single_optimization()
        initval = cur_initval
        vstar = vstar_cur
        if (abs(cdf_fixed(vstar) - quant) <= 0.05)
            break        
        end
        count = count + 1;
    end
    if count >= 2
        @warn "number restarts is", count
    end
    if (abs(cdf_fixed(vstar) - quant) > 0.05)
        error("Optimization step did not converge")
    end
    return vstar, initval, cdf_fixed(vstar)
end
