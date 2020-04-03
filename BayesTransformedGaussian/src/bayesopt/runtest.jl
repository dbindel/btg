include("./kernel_system.jl")

capacity = 100
d, p = 2, 5
n = 10

k = FixedParam(Gaussian(), 2)

U, Fu, X, Fx = sdata_structs(capacity, d, p)
Ku, Kux, rKx = kdata_structs(capacity, d, p)

U .= rand(d, p)
Fu .= rand(p, p)

y = rand(p+n)
@views yu, yx = y[1:p], y[p+1:end]

ks = kernel_system!(Ku, Kux, rKx, Fu, Fx, k, U, X)
println(size(ks.sd.X.A), ks.sd.X.n)
X′, Fx′ = extend(ks, n)
X′ = rand(d, n)
Fx′ = rand(p, n)
update!(ks, n, X′, Fx′)

c = similar(y)
cu, cx = solve!(c, ks, y)
d = similar(cu)
fit_tail!(d, ks, yu, cu, cx)

display(c)
display(d)
