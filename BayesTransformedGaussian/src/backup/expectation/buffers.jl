@doc raw"""
"""
struct PrecomputeTheta{T}
    matΣ::Array{Float64, 3}
    cholΣ::Vector{T}
    ΣX::Matrix{Float64}
    XΣX::Matrix{Float64}
end

@doc raw"""
"""
struct PrecomputeBuffer
    β::Vector{Float64}
    q::Vector{Float64}
    Σg::Vector{Float64}
    pθλ::Float64
end

@doc raw"""
"""
struct Buffer
    B::Vector{Float64}
    ΣB::Vector{Float64}
    D::Float64
    H::Vector{Float64}
    C::Float64
    m::Float64
end
