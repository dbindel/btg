
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
    buf::B
end

function compute_dists!(btg::BTG)
end

function compute_dists_θλ!(btg::BTG)
end

function compute_dists_λ!(btg::BTG)
end
