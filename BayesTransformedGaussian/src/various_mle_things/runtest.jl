include("./kernel_system.jl")

function test_data(d, p, m, capacity)
    U = rand(d, p)
    Fu = rand(p, p)
    X = data_array(rand(d, capacity), m)
    Fx = data_array(rand(p, capacity), m)
    y = data_array(rand(capacity), p + m)

    return U, Fu, X, Fx, y
end

function test_const(U, Fu, X, Fx, capacity)
    d, p = size(U)
    m = size(X, 2)
    k = FixedParam(Gaussian(), 0.01)
    K = Matrix{Float64}(undef, m + p, m + p)
    correlation!(K, k, [U X])
    M = [zeros(p, p) [Fu Fx]; [Fu Fx]' K]
    sd = system_data!(U, Fu, X, Fx)
    Ku = Matrix{Float64}(undef, p, p)
    Kux = data_array(Float64, capacity, p)
    Kx = incremental_cholesky(Float64, capacity)
    kd = kernel_data!(k, Ku, Kux, Kx, sd.U)
    compute_next!(kd, sd)
    ks = KernelSystem(sd, kd)

    return ks, M
end

function test_solve(ks, M, y)
    cu, cx = solve!(similar(y), ks, y)
    d = fit_tail!(similar(cu), ks, y[1:length(cu)], cu, cx)
    c = M \ [zeros(length(cu)); y]
    display([d; cu; cx])
    display(c)
    return nothing
end

function test_system()
end
