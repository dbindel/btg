@doc raw"""
"""
struct PrecomputeTheta
    matΣ::Matrix{Float64}
    cholΣ::Cholesky{Float64, Matrix{Float64}}
    ΣX::Vector{Float64}
end

@doc raw"""
"""
struct PrecomputeBuffer
    β::Vector{Float64}
    q::Vector{Float64}
    pθλ::Float64
end

@doc raw"""
"""
struct Buffer
    D::Float64
    H::Vector{Float64}
    C::Float64
    m::Float64
end
