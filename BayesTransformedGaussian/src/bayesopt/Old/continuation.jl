include("./incremental.jl")
include("./kernel.jl")
include("./kernel_system.jl")

using Plots

function get_out_of_bounds(y, ℓ, u)
    for i in length(y)
        if y[i] > u[i] + eps() * 10^4 
            return i, u[i]
        elseif y[i] < ℓ[i] - eps() * 10^4
            return i, ℓ[i]
        end
    end
    return 0
end

function eval!(y, kd, y1, y2, x)
    for i in size(x, 2)
        y[i] = predict_point(kd, y1, y2, x[:, i], ones(1, 1))
    end
    return nothing
end

function active_set(k, x, y, ℓ, u)
    n = size(x, 2)
    kd = kernel_data(n, k, x[:, 1:1], ones(1, 1), x[:, n:n], ones(1, 1))
    active = zeros(Bool, n)
    active[1] = active[n] = true
    yhat = Array{Float64}(undef, n)
    eval!(yhat[.!active], kd, y[1], y[2:end][active[2:end]], x[:, .!active])
    out_of_bounds = get_out_of_bounds(yhat[.!active], ℓ[.!active], u[.!active])
    
    while out_of_bounds[1] >= 0
        i = out_of_bounds[1]
        add_point!(kd, x[:, .!active][:, i], ones(1))
        y[.!active][i] = out_of_bounds[2]
        active[.!active][i] = true
        eval!(yhat[.!active], kd, y[1], y[2:end][active[2:end]], x[:, .!active])
        out_of_bounds = get_out_of_bounds(yhat, ℓ, u)
    end
    return active, yhat
end

k = FixedParam(RBF(), 0.05)
x = collect(-1:0.1:1)
x = reshape(x, 1, length(x))

f(t) = exp(-t ^ 2) * cospi(2t)
y = f.(x)
y = reshape(y, length(y))
plt = plot([x' y])
savefig(plt, "base.png")

ℓ = y .- 10
u = y .+ 10
active, yhat = active_set(k, x, y, ℓ, u)
plot!(plt, [x yhat])
scatter!(plt, [x[active] yhat[active]])
savefig(plt, "plot1.png")

ℓ = y .- 0.2
u = y .+ 0.2
active, yhat = active_set(k, x, y, ℓ, u)
plot!(plt, [x yhat])
scatter!(plt, [x[active] yhat[active]])
savefig(plt, "plot2.png")

ℓ = y .- 0.1
u = y .+ 0.1
active, yhat = active_set(k, x, y, ℓ, u)
plot!(plt, [x yhat])
scatter!(plt, [x[active] yhat[active]])
savefig(plt, "plot3.png")
