using Ipopt, JuMP
import JuMP.MathOptInterface
const MOI = MathOptInterface

my_square(x) = x^2
my_square_prime(x) = 2x
my_square_prime_prime(x) = 2

function wrapper(g, x)
    ∇my_g(g, x...)
end

#∇my_g(g, [2, 3]...)
function ∇my_g(g, x...)
    args = x
    @info "args", args
    g[1] = args[2]^2
    g[2] = 2*args[1]args[2]
end;
#∇h(g, [1, 2]...)

my_f(x, y) = (x - 1)^2 + (y - 2)^2
function ∇f(g, x, y)
    g[1] = 2 * (x - 1)
    g[2] = 2 * (y - 2)
end;

model = Model(Ipopt.Optimizer)
#m = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
#register(model, :my_f, 2, my_f, ∇f)
register(model, :my_f, 2, my_f, autodiff=true)
register(model, :my_square, 1, my_square, my_square_prime,
         my_square_prime_prime)
@variable(model, 1.2 >= varname[1:2] >= 0.501)
@NLobjective(model, Min, my_f(varname[1], my_square(varname[2])))
@NLconstraint(model, log(varname[1] + varname[2]) == log(2))

#@NLconstraint(model, x[1]*my_square(x[2]) == 0)
JuMP.optimize!(model)
println(value.(varname))

