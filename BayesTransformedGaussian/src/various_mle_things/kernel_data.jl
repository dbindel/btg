#####
##### Dependencies
#####


import Base.display

#####
##### Structs
#####

@doc raw"""
"""
struct KernelData{K<:AbstractCorrelation}
    kern::K # Kernel Correlation function
    Ku::Matrix{Float64} # Correlation at locations U
    Kux::DataMatrix{Float64} # Cross correlation of observation locations
    rKx::IncrementalCholesky{Float64} # Reduced kernel system at locations X
end

#####
##### Construction
#####

function kernel_data!(kern, Ku, Kux, rKx, U)
    correlation!(Ku, kern, U)
    return KernelData(kern, Ku, Kux, rKx)
end

function valid_update(kd::KernelData, k)
    return valid_update(kd.Kux, k) && valid_update(kd.rKx, k)
end

function view_next(kd::KernelData, sd)
    k = size(sd.X, 2) - size(kd.Kux, 2)
    valid_update(kd, k) || throw(ErrorException("Size cannot exceed capacity"))
    Kxx′, Kx′ = view_next(kd.rKx, k)
    return (Kux′ = view_next(kd.Kux, k), Kxx′ = Kxx′, Kx = Kx′)
end

function compute_next!(kd::KernelData, sd)
    n = size(kd.Kux, 2)
    k = size(sd.X, 2) - n
    @views X, X′ = sd.X[:, 1:n], sd.X[:, n+1:end]
    @views W, W′ = sd.W[:, 1:n], sd.W[:, n+1:end]
    Kux′, Kxx′, Kx′ = view_next(kd, sd)
    
    cross_correlation!(Kux′, kd.kern, sd.U, X′)
    cross_correlation!(Kxx′, kd.kern, X, X′)
    correlation!(Kx′, kd.kern, X′)

    reduced_update!(Kxx′, Kx′, kd.Ku, kd.Kux, Kux′, W, W′)

    compute_next!(kd.Kux, k)
    compute_next!(kd.rKx, k)
    return nothing
end

function remove_last!(kd::KernelData, sd)
    k = size(kd.Kux, 2) - size(sd.X, 2)
    valid_update(kd, -k) || throw(ErrorException("Size cannot be negative"))
    remove_last!(kd.Kux, k)
    remove_last!(kd.rKx, k)
    return nothing
end

function reset!(kd::KernelData)
    remove_last!(kd.Kux, size(kd.Kux, 2))
    remove_last!(kd.rKx, size(kd.rKx, 2))
    return nothing
end

#####
##### Functions
#####

function Base.display(kd::KernelData)
    print("\n    Correlation Kernel:\n")
    display(kd.kern)
    println("\n    Correlation of U points:\n")
    display(kd.Ku)
    println("\n    Cross correlation between U and X:\n")
    display(kd.Kux)
    println("\n    Solver for the reduced kernel system:\n")
    display(kd.rKx)
    return nothing
end

function reduced_update!(Kxx′, Kx′, Ku, Kux, Kux′, W, W′)
    tmp = Kux′ .- Ku * W′ 
    
    mul!(Kxx′, W', tmp, -1, 1)
    mul!(Kxx′, Kux', W′, -1, 1)
    mul!(Kx′, (W′)', tmp, -1, 1)
    mul!(Kx′, (Kux′)', W′, -1, 1)
    
    return Kxx′, Kx′
end

function diag_reduced_schur!(Dx′, Kxx′, kd, Kux′, W, W′)
    tmp = Kux′ .- kd.Ku * W′
    
    mul!(Kxx′, W', tmp, -1, 1)
    mul!(Kxx′, kd.Kux', W′, -1, 1)
    ldiv!(get_chol(kd.rKx).L, Kxx′)
    
    for i in axes(Dx′, 1)
        @views Dx′[i] = 1 - dot(W′[:, i], tmp[:, i]) - dot(Kux′[:, i], W′[:, i])
    end
    
    r = similar(Dx′, 1, length(Dx′))
    sum!(abs2, r, Kxx′)
    @views Dx′ .-= r[:]
    return Dx′
end
