include("./kernel.jl")
include("./transform.jl")
include("./incremental.jl")

#####
##### Structs
#####

@doc raw"""
"""
mutable struct SystemData
    U::Matrix{Float64} # p x p subset of locations with unisolvent F
    Fu::LU{Float64, Matrix{Float64}} # unisolvent covariates at U
    X::DataMatrix{Float64} # Observation locations
    W::DataMatrix{Float64} # Fu \ Fx
end

@doc raw"""
"""
struct KernelData{K<:AbstractCorrelation}
    k::K # Kernel Correlation function
    Ku::Matrix{Float64} # Correlation at locations U
    Kux::DataMatrix{Float64} # Cross correlation of observation locations
    rKx::IncrementalCholesky{Float64} # Reduced kernel system at locations X
end

@doc raw"""
"""
struct KernelSystem{K<:AbstractCorrelation}
    sd::SystemData
    kd::KernelData{K}
end

#####
##### Construction
#####

function sdata_structs(capacity, d, p)
    U = Matrix{Float64}(undef, d, p)
    Fu = Matrix{Float64}(undef, p, p)
    X = data_array(Matrix{Float64}(undef, d, capacity), 0)
    Fx = data_array(Matrix{Float64}(undef, p, capacity), 0)
    return U, Fu, X, Fx
end
nothing
function kdata_structs(capacity, d, p)
    Ku = Matrix{Float64}(undef, p, p)
    Kux = data_array(Matrix{Float64}(undef, p, capacity), 0)
    rKx = incremental_cholesky!(Matrix{Float64}(undef, capacity, capacity), 0)
    return Ku, Kux, rKx
end

function kernel_system!(Ku, Kux, rKx, Fu, Fx, k, U, X)
    sd = system_data!(Fu, Fx, U, X)
    return KernelSystem(sd, kernel_data!(Ku, Kux, rKx, k, sd))
end

extend(ks::KernelSystem, k) = extend(ks.sd, k)
function update!(ks::KernelSystem, k, X, Fx)
    ldiv!(ks.sd.Fu, Fx)
    update!(ks.kd, ks.sd, X, Fx)
    update!(ks.sd, k)
    return nothing
end

function system_data!(Fu, Fx, U, X)
    # TODO error checking for input dimensions
    Fu_fact = lu!(Fu)
    ldiv!(Fu_fact, array_view(Fx))
    return SystemData(U, Fu_fact, X, Fx) 
end

function valid_update(sd::SystemData, k)
    return valid_update(sd.X, k) && valid_update(sd.W, k)
end

function extend(sd::SystemData, k)
    valid_update(sd, k) || throw(ErrorException("Size cannot exceed capacity"))
    return (X = extend(sd.X, k), Fx = extend(sd.W, k))
end
function update!(sd::SystemData, k)
    valid_update(sd, k) || throw(ErrorException("Size cannot exceed capacity"))
    update!(sd.X, k)
    update!(sd.W, k)
    return nothing
end
function remove!(sd::SystemData, k)
    valid_update(sd, -k) || throw(ErrorException("Size cannot be negative"))
    remove!(sd.X, k)
    remove!(sd.W, k)
    return nothing
end

function system_schur!(Kxx′, Kx′, Ku, Kux, Kux′, W, W′)
    mul!(Kxx′, W', Kux′, -1, 1)
    mul!(Kxx′, Kux', W′, -1, 1)
    
    mul!(Kx′, W′', Kux′, -1, 1)
    mul!(Kx′, Kux′', W′, -1, 1)

    tmp = Ku * W′
    mul!(Kxx′, W', tmp, 1, 1)
    mul!(Kx′, W′', tmp, 1, 1)
    return nothing
end

function kernel_data!(Ku, Kux, rKx, k, sd)
    # TODO error checking for input dimensions
    correlation!(Ku, k, sd.U)
    cross_correlation!(Kux, k, sd.U, sd.X)

    p, m = size(sd.W)
    Kxx′, Kx′ = extend(rKx, m)
    correlation!(Kx′, k, sd.X)
    system_schur!(Kxx′, Kx′, Ku, ones(p, 0), Kux, ones(p, 0), sd.W) 
    update!(rKx, m)
    
    return KernelData(k, Ku, Kux, rKx)
end

function valid_update(kd::KernelData, k)
    return valid_update(kd.Kux, k) && valid_update(kd.rKx, k)
end

function extend(kd::KernelData, k)
    valid_update(kd, k) || throw(ErrorException("Size cannot exceed capacity"))
    Kxx′, Kx′ = extend(kd.rKx, k)
    return (Kux′ = extend(kd.Kux, k), Kxx′ = Kxx′, Kx′ = Kx′)
end
function update!(kd::KernelData, sd, X′, W′)
    p, k = size(W′)
    Kux′, Kxx′, Kx′ = extend(kd, k)
    
    cross_correlation!(Kux′, kd.k, sd.U, X′)
    cross_correlation!(Kxx′, kd.k, sd.X, X′)
    correlation!(Kx′, kd.k, X′)
    
    system_schur!(Kxx′, Kx′, kd.Ku, kd.Kux, Kux′, sd.W, W′)
    update!(kd.Kux, k)
    update!(kd.rKx, k)
    return nothing
end
function remove!(kd::KernelData, k)
    valid_update(kd, -k) || throw(ErrorException("Size cannot be negative"))
    remove!(kd.Kux, k)
    remove!(kd.rKx, k)
    return nothing
end

#####
##### Functions
#####

function reduced_solve!(cx, ks, yu, yx)
    cx .= yx
    mul!(cx, ks.sd.W', yu, -1, 1)
    ldiv!(ks.kd.rKx, cx)
    return nothing
end
function solve!(c, ks, y)
    p = size(ks.sd.W, 1)
    @views cu, cx = c[1:p], c[p+1:end]
    @views yu, yx = y[1:p], y[p+1:end]
    reduced_solve!(cx, ks, yu, yx)
    mul!(cu, ks.sd.W, cx)
    return cu, cx
end
function fit_tail!(d, ks, yu, cu, cx)
    d .= yu
    mul!(d, ks.kd.Ku, cu, -1, 1)
    mul!(d, ks.kd.Kux, cx, -1, 1)
    ldiv!(ks.sd.Fu', d)
    return nothing
end

#####
##### Prediction & Statistics
#####

function det(ks::KernelSystem)
    d, s = logabsdet(ks)
    return s * exp(d)
end
function logdet(ks::KernelSystem)
    d, s = logabsdet(ks)
    s < 0 || throw(DomainError(s))
    return d
end
function logabsdet(ks::KernelSystem)
    s = (-1) ^ (size(ks.sd.Fu, 1) % 2)
    return 2 * logabsdet(ks.sd.Fu)[1] + logdet(ks.kd.rKx), s
end

function predict!(ks::KernelSystem, y, X′, Fx′)
    cu, cx = solve!(similar(y), ks, y)
    d = similar(cu)
    fit_tail!(d, ks, @views y[1:length(d)], cu, cx)
end

function predict_point(kd::KernelData, y, X′, Fx′)
    c1, c2, β = solve_system(kd, y1, y2)
    k1 = Array{Float64}(undef, size(kd.K1, 1))
    k2 = Array{Float64}(undef, kd.n)
    point_correlation!(k1, k2, kd, x)
    m = dot(k1, c1) + dot(k2, c2) + dot(fx, β)
    return m
end

function qmC!(kd::KernelData, y1, y2, x, fx)
    c1, c2, β = solve_system(kd, y1, y2)
    k1 = Array{Float64}(undef, size(kd.K1, 1))
    k2 = Array{Float64}(undef, kd.n)
    point_correlation!(k1, k2, kd, x)
    m = dot(k1, c1) + dot(k2, c2) + dot(fx, β)
    
    q = dot(y1, c1) + dot(y2, c2)
    
    add_point!(kd, x, fx)
    C = kd.K2_tilde.R[kd.n, kd.n] ^ 2
    remove_point!(kd)
    
    return q, m, C
end
