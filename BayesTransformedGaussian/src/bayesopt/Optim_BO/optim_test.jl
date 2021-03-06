using Optim
using PyPlot

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

optimize(f, [0.0, 0.0], NelderMead())

function g!(storage, x)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

res = optimize(f, g!, [0.0, 0.0], LBFGS())
 
figure(1)
x = collect(Float16, range(-1,length=100,stop=1));
y = collect(Float16, range(-1,length=100, stop=1));
z = hcat(x, y); z = mapslices(f, z, dims = 2)
surf(x,y,z);

