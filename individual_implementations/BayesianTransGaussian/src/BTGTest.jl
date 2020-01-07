using Distributions

using Profile
using BenchmarkTools

X = 40 * rand(30, 1) .+ 20;
Y = (2 * X .+ randn(30, 1)).^0.5
Y = reshape(Y, length(Y))

k = SquaredExponential(Uniform());
g = BoxCox(Uniform(1, 3))
f = Identity()
mdl = Model(k, g, f, X, Y, 30, 30);

x = [20.0]

show(equalinterval(mdl, 0.95, x))
display(plotdensity(mdl, 0.95, x, 20))

@benchmark equalinterval(mdl, 0.95, x)

