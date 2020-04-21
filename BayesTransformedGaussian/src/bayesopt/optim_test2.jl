using Optim
using PyPlot
using Distances
###
### Box Constrained Minimization
###

const charges = [[0, 0]] #fixed charges

function lennardJones(x, y)
    r = SqEuclidean()(x, y)
    return (1-2*(r+1)^6)/(r+1)^12
end

function potential(charges::Array{Array{T, 1}, 1} where T<:Real)
    function f(x)
        s = 0
        for c in charges
            s = s + lennardJones(x, c) 
        end
        return s
    end
    return f
end

pfun = potential(charges)
boundedpfun = x -> (val = pfun(x); val > 5.0 ? 5.0 : val)  

figure(2)
x = collect(Float64, range(-5, length = 500, stop = 5))
y = collect(Float64, range(-5, length = 500, stop = 5))
z = hcat(x, y); z = mapslices(boundedpfun, z, dims = 1)
surf(x, y, z)


if false 
func(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

figure(1)
x = collect(Float16, range(-1,length=100,stop=1));
y = collect(Float16, range(-1,length=100, stop=1));
z = hcat(x, y); z = mapslices(func, z, dims = 2)
surf(x,y,z);
scatter3D([0], [0], [100], color = "red")

function fun_grad!(g, x)
g[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
g[2] = 200.0 * (x[2] - x[1]^2)
end

function fun_hess!(h, x)
h[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
h[1, 2] = -400.0 * x[1]
h[2, 1] = -400.0 * x[1]
h[2, 2] = 200.0
end;

x0 = [0.0, 0.0]
df = TwiceDifferentiable(func, fun_grad!, fun_hess!, x0)

lx = [-0.5, -0.5]; ux = [0.5, 0.5]
dfc = TwiceDifferentiableConstraints(lx, ux)

res = optimize(df, dfc, x0, IPNewton())

end
