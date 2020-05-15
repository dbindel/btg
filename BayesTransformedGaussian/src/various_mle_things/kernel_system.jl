#####
##### Dependencies
#####

include("./kernel.jl")
include("./transform.jl")
include("./incremental.jl")
include("./system_data.jl")
include("./kernel_data.jl")

import LinearAlgebra: logabsdet

#####
##### Structs
#####

@doc raw"""
"""
struct KernelSystem{K<:AbstractCorrelation}
    sd::SystemData
    kd::KernelData{K}
end

#####
##### Construction
#####


#####
##### Functions
#####

function reduced_solve!(cx, ks, yu, yx)
    cx .= yx
    mul!(cx, ks.sd.W', yu, -1, 1)
    ldiv!(ks.kd.rKx, cx)
    return cx
end
function solve!(c, ks, y)
    p = size(ks.sd.W, 1)
    @views cu, cx = c[1:p], c[p+1:end]
    @views yu, yx = y[1:p], y[p+1:end]
    reduced_solve!(cx, ks, yu, yx)
    mul!(cu, ks.sd.W, cx, -1, 0)
    return cu, cx
end
function fit_tail!(d, ks, yu, cu, cx)
    d .= yu
    mul!(d, ks.kd.Ku, cu, -1, 1)
    mul!(d, ks.kd.Kux, cx, -1, 1)
    ldiv!(ks.sd.Fu', d)
    return d
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
    s >= 0 || throw(DomainError(s))
    return d
end
function logabsdet(ks::KernelSystem)
    s = (-1.0) ^ (size(ks.sd.U, 1) % 2)
    return 2 * logabsdet(ks.sd.Fu)[1] + logdet(ks.kd.rKx), s
end

function predict!(m, ks::KernelSystem, cu, cx, d, X′, Fx′) # this is m
    Kux′ = similar(Fx′)
    Kxx′ = Array{Float64}(undef, size(ks.sd.X, 2), size(X′, 2))
    
    cross_correlation!(Kux′, ks.kd.kern, ks.sd.U, X′)
    cross_correlation!(Kxx′, ks.kd.kern, ks.sd.X, X′)

    mul!(m, Kux′', cu, 1, 1)
    mul!(m, Kxx′', cx, 1, 1)
    mul!(m, Fx′', d, 1, 1)
    
    return m
end

 # This is q
energy(cu, cx, y) = @views dot(cu, y[1:length(cu)]) + dot(cx, y[length(cu)+1:end])

# Returns vectors m, (qC)^-1 for test points
function predict_dist!(m, σ, ks::KernelSystem, cu, cx, d, q, X′, Fx′)
    W′ = similar(Fx′)
    Kux′ = Array{Float64}(undef, size(Fx′)...)
    Kxx′ = Array{Float64}(undef, size(ks.sd.X, 2), size(X′, 2))
    Dx′ = similar(σ)

    ldiv!(W′, ks.sd.Fu, Fx′)
    cross_correlation!(Kux′, ks.kd.kern, ks.sd.U, X′)
    cross_correlation!(Kxx′, ks.kd.kern, ks.sd.X, X′)

    mul!(m, Kux′', cu, 1, 1)
    mul!(m, Kxx′', cx, 1, 1)
    mul!(m, Fx′', d, 1, 1)

    diag_reduced_schur!(Dx′, Kxx′, ks.kd, Kux′, ks.sd.W, W′)
    σ .= inv.(Dx′ .* q)
    
    return m, σ
end
