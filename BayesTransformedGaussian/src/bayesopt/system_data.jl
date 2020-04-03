#####
##### Dependencies
#####

import Base: display

#####
##### Structs
#####

@doc raw"""
"""
struct SystemData
    U::Matrix{Float64} # p x p subset of locations with unisolvent F
    Fu::LU{Float64, Matrix{Float64}} # unisolvent covariates at U
    X::DataMatrix{Float64} # Observation locations
    W::DataMatrix{Float64} # Fu \ Fx
end

#####
##### Construction
#####

function system_data!(U, Fu, X, Fx)
    # TODO error checking for input dimensions
    Fu_fact = lu!(Fu)
    ldiv!(Fu_fact, array_view(Fx))
    return SystemData(U, Fu_fact, X, Fx)
end

function valid_update(sd::SystemData, k)
    return valid_update(sd.X, k) && valid_update(sd.W, k)
end

function view_next(sd::SystemData, k)
    valid_update(sd, k) || throw(ErrorException("Size cannot exceed capacity"))
    return (X = view_next(sd.X, k), Fx = view_next(sd.W, k))
end

function compute_next!(sd::SystemData, k)
    valid_update(sd, k) || throw(ErrorException("Size cannot exceed capacity"))
    compute_next!(sd.X, k)
    ldiv!(sd.Fu, view_next(sd, k).Fx)
    compute_next!(sd.W, k)
    return nothing
end

function remove_last!(sd::SystemData, k)
    valid_update(sd, -k) || throw(ErrorException("Size cannot be negative"))
    remove_last!(sd.X, k)
    remove_last!(sd.W, k)
    return nothing
end

function reset!(sd::SystemData)
    remove_last!(sd.X, size(sd.X, 2))
    remove_last!(sd.W, size(sd.W, 2))
    return nothing
end

#####
##### Functions
#####

function Base.display(sd::SystemData)
    println("\n    U observations:\n")
    display(sd.U)
    println("\n    Fu factorization:\n")
    display(sd.Fu)
    println("\n    X observations:\n")
    display(sd.X)
    println("\n    Fu \\ Fx:\n")
    display(sd.W)
    return nothing
end
