using Random

@doc raw"""
"""
abstract type AbstractQuadrature end

@doc raw"""
"""
mutable struct DiscreteQuadrature{T}
    n::Int
    idx::Int
    weights::Vector{T}
    nodes::Vector{T}
    seed::MersenneTwister
    ordered::Bool
end

function next!(quad::DiscreteQuadrature)
    if quad.ordered
        ret = (quad.nodes[quad.idx], quad.weights[quad.idx])
    else
        i = rand(quad.seed, Int) % quad.n
        ret = (quad.nodes[i], quad.weights[i])
    end
    quad.idx = quad.idx % quad.n + 1
    return ret
end


@doc raw"""
"""
struct RandomQuadrature{D<:Distribution}
    n::Int
    idx::Int
    dist::D
    seed::MersenneTwister
end

function next!(quad::ContinuousQuadrature)
    ret = (rand(
    quad.idx = quad.idx % quad.n + 1
    return ret
end
