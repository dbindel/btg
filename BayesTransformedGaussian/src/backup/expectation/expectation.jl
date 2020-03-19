
@doc raw"""
"""
abstract type QuadratureRule end

@doc raw"""
    Parameterized
"""
struct Parameterized{C<:Function, T<:Tuple{Vararg{Distribution}}, Q<:QuadratureRule}
    constructor::C
    priors::T
    quadrature::Q
end

@doc raw"""
    Buffer
"""
struct Buffer{T<:Tuple{Vararg{AbstractArray}}}
    bufs::T
end

@doc raw"""
    GaussQuad
"""
struct GaussQuad end

@doc raw"""
    TODO Unimplimented
"""
struct SparseQuad end

@doc raw"""
    TODO Unimplimented
"""
struct MonteCarloQuad end

