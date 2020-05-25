include("../btg.jl")
include("btgBayesopt.jl")
#include("../computation/finitedifference.jl")
using Plots
using Random
Random.seed!(8);

#####
##### Define optimization problem, initialize GP with burn-in points, initialize btg parameters
#####
Himmelblau(x) = (x[1]^2 + x[2] -11)^2 + (x[1]+x[2]^2-7)^2 #function to optimize
y, x = sample_points(Himmelblau, [-5, -5], [5, 5]; num = 10) #burn-in points which are used as training data in GP
x = vcat(x, [5 5])
y = vcat(y, 890)
x = vcat(x, [-4.999 4.999])
y = vcat(y, 530)
x = vcat(x, [4.999 -4.999])
y = vcat(y, 610)
x = vcat(x, [-4.999 -4.999])
y = vcat(y, 249.6042059640019)
x = vcat(x, [5 2.55573])
y = vcat(y, 294.57103540062496)
x = vcat(x, [5.0 2.3475172404355718])
y = vcat(y, 279.5672977301108)

cur_min = minimum(mapslices(Himmelblau, x, dims = 2))

@info "y-vals", y[1:5] 
### Set BTG parameters and get function handles for pdf, cdf, dpdf, etc. 
Fx = linear_polynomial_basis(x)
train = extensible_trainingData(x, Fx, y) #training data triple: location, covariates, labels
#rangeθ = reshape(select_single_theta_range(x), 1, 2) #rangetheta is auto-selected
#rangeθ = [0.0001 1000]
#rangeθ = [10.0 300] 
rangeθ  = [50.0 1000.0]
@info "rangeθ", rangeθ 
rangeλ = [-1.0 1] 
btg1 = btg(train, rangeθ, rangeλ);
 #get function handles for pdf, cdf, derivtives
lx = [1, -5, -5]; ux = [3000.0, 5, 5] 
#(_, _, plt1, pol) = checkDerivative(cdf_fixed, cdf_gradient_fixed, vstars[1],  nothing, 4, 8)

"""
Takes n BO steps using a pre-initialized btg1 object, which already contains some points in its kernel system
Returns log of points chosen from optimizing UCB optimization problem

INPUTS: f, expensive function we are optimizing with BO
 - min: minimum value of f on points already inside BO kernel system (we assume it's initialized with burn-in points)
 
NOTE: assumes linear polynomial basis, e.g. covariates for point [x, y] is [1, x, y] 

OUTPUTS:
    -  x_hist: history of places what the GP chooses to sample (via optimizing the acquisition function)
    -  f_hist: history of actual function evaluations at x_hist
    -  gp_hist: history of values the gp think is at 25th quantile
    -  min_hist: keeps track of minimum over all previous f_evals, including burn-in
    -  init_hist: history of initializations for acqusition function optimization routines
    -  quant_hist: history of cdf_evals at chosen sample points (should be 0.25)
"""
function BO(btg1::btg, f, n, lx, ux, min) #BO loop
    x_hist = [] #keep track of selected points
    f_hist = []
    gp_hist = [] #compare what the gp thinks the objective looks like to what it actually is (f_hist)
    min_hist = [min]
    init_hist = []
    quant_hist = []
    #debug histories
    training_size_hist = []
    argmin_hist = [x[argmin(getLabel(btg1.trainingData)):argmin(getLabel(btg1.trainingData)), :]]

    for i = 1:n #take 10 BO steps
        println("========================ITERATION: ", i)
        (_, cur_cdf, _, cdf_gradient) = solve(btg1; derivatives = true);
        #(vstar, initval, quant_eval) = optimize_acqusition(cdf, cdf_gradient, lx, ux; maxiter = 300, initial = argmin_hist[end]);
        (vstar, initval, quant_eval) = optimize_acqusition(cur_cdf, cdf_gradient, lx, ux; maxiter = 300, initial = [], quant = 0.5);
        s_star = reshape(vstar[2:end], 1, length(vstar[2:end]))
        push!(quant_hist, quant_eval)
        push!(init_hist, initval[2:end])
        push!(gp_hist, vstar[1])
        push!(x_hist, s_star) #save
        push!(training_size_hist, btg1.trainingData.n)
        f_star = f(s_star)
        push!(f_hist, f_star)
        if f_star < min_hist[end]
            push!(min_hist, f_star)
            push!(argmin_hist, s_star)
        else
            push!(min_hist, min_hist[end])
            push!(argmin_hist, argmin_hist[end])
        end
        #@info "f_star", f_star
        updateBTG!(btg1, s_star, linear_polynomial_basis(s_star), [f_star]) #update with chosen point 
    end
    return x_hist, f_hist, min_hist, gp_hist, init_hist, quant_hist, training_size_hist
end    
(x_hist, f_hist, min_hist, gp_hist, init_hist, quant_hist, training_size_hist) = BO(btg1, Himmelblau, 5, lx, ux, cur_min);


include("analyze.jl")