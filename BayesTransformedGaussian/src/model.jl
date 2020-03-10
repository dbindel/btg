using Distances, StaticArrays, StatsFuns

@doc raw"""
    BTG
"""
mutable struct BTG{
    T<:Real,
    M<:AbstractMatrix{T},
    V<:AbstractVector{T},
    G<:Parameterized{},
    K<:Parameterized{},
    B<:Buffer{}
}
    S0::M
    X0::M
    Y0::V
    g::G
    k::K
    preθ::PrecomputeTheta
    preBuf::PrecomputeBuffer
    buf::Buffer
end

function compute_dists!(
    btg::BTG,
    s::Vector{Float64},
    x::Vector{Float64}
    y::Float64
)
    ret = @MVector [0.0, 0.0, 0.0]
    compute_dists_θλ!(ret, btg, s, x, y)
    reset!(btg.g)
    reset!(btg.k)
    ret ./ btg.preBuf.normalizing_constant
    return ret
end

function compute_dists_θλ!(
    ret::SVector{3, Float64},
    btg::BTG,
    s::Vector{Float64},
    x::Vector{Float64}
    y::Float64
)
    for i in 1:num_nodes(btg.k) # TODO
        colwise!(btg.buf.B, SqEuclidean(), btg.S0, s)
        btg.buf.B .= btg.k.(btg.buf.B)
        ldiv!(btg.buf.ΣB, btg.preθ.cholΣ[i], btg.buf.B)

        btg.buf.D = 1 - dot(btg.buf.B, btg.buf.ΣB)
        btg.buf.H .= x .- btg.X0' * btg.buf.ΣB # TODO Buffer for this?
        btg.buf.C = btg.buf.D + btg.buf.H' * (btg.preθ.XΣX \ btg.buf.H) # TODO another chol vec for this
        tmp = @MVector [0.0, 0.0, 0.0]
        compute_dists_λ!(tmp, btg, s, x, y)
        ret += btg.k.weight * tmp
        next!(btg.k)
        reset!(btg.g)
    end
    return nothing
end

function compute_dists_λ!(
    ret::SVector{3, Float64},
    btg::BTG,
    s::Vector{Float64},
    x::Vector{Float64}
    y::Float64
)
    for j in 1:num_nodes(btg.g)
        btg.buf.m .= dot(btg.buf.B, btg.preBuf.Σg)
        p = (y - btg.buf.m) / btg.buf.q / btg.buf.C
        ν = size(btg.X0, 1) - size(btg.X0, 2)
        ret += btg.g.weight * btg.preBuf.pθλ * (@SVector [tdistcdf(ν, p), tdistpdf(ν, p), 0.0]) # Add Leo's derivative
        next!(btg.g)
    end
    return nothing
end
