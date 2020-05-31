import Base: inv

"""
Family of nonlinear transformation parametrized by λ that defaults to Box Cox.
N.B. We assume in derivatives.jl that the nonlinear transformation f is monotonic increasing. 
"""

@doc raw"""
    BoxCox(λ::Real)

The Box-Cox power transformation function with parameter λ.

```math
g_\lambda(x) = \begin{cases} 
    \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0\\ 
    \ln(x) & \lambda = 0
\end{cases}
```

External links
* [Power Transform on Wikipedia](https://en.wikipedia.org/wiki/Power_transform)

...
# Arguments
* `x::float64`: evaluation point
* `lambda::float=1.0`: lambda hyperparameter

"""
abstract type NonlinearTransform end

struct BoxCox <:NonlinearTransform end 

struct anotherone<:NonlinearTransform end

#jitter = 1e-6
(::BoxCox)(x, λ::Union{Array{T, 1}, T} where T<:Real) = (@assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real); 
                                                 λ[1] == 0 ? log.(x ) : expm1.(log.(x) .* λ[1]) ./ λ[1])
partialx(::BoxCox, x, λ) =  λ[1]==0 ? float(x).^(-1) : float(x).^(λ[1] .-1)
partialxx(::BoxCox, x, λ) = λ[1]==0 ? - float(x).^(- 2) : (λ[1] .- 1) * float(x) .^ (λ[1] .- 2)
partialλ(::BoxCox, x, λ) = λ[1]==0 ? 0 : (λ[1] .* float(x).^λ[1] .* log.(x) .- float(x).^λ[1] .+ 1)/λ^2
partialλλ(::BoxCox, x, λ) = (λ[1]==0 ? 0 : float(x).^λ[1] .* (Base.log.(x)^2)/λ[1] - 
                            (2 * x.^λ[1] .* Base.log.(x))/λ[1]^2 + 2*(float(x).^λ[1] - 1)/λ[1]^3)
partialxλ(::BoxCox, x, λ) = λ[1]==0 ? 0 : float(x).^(λ[1] -1) .* Base.log.(x)
inverse(::BoxCox, x, λ) = λ[1]==0 ? Base.exp.(x) : Base.exp.(Base.log.(λ[1].*x.+1)./λ[1])

struct ShiftedBoxCox <:NonlinearTransform end 
(::ShiftedBoxCox)(x, λ::Union{Array{T, 1}, T} where T<:Real) = (@assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real); 
                                                 λ[1] == 0 ? log.(x .+ 1) : expm1.(log.(x .+ 1) .* λ[1]) ./ λ[1])
partialx(::ShiftedBoxCox, x, λ) =  λ[1]==0 ? float(x .+ 1).^(-1) : float(x .+ 1).^(λ[1] .-1)
partialxx(::ShiftedBoxCox, x, λ) = λ[1]==0 ? - float(x .+ 1).^(- 2) : (λ[1] .- 1) * float(x .+ 1) .^ (λ[1] .- 2)

struct YeoJohnson <:NonlinearTransform end
function (::YeoJohnson)(x, λ::Union{Array{T, 1}, T}) where T<:Real
    @assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real)
    if x >= 0
        g = BoxCox()
        return g(x+1,  λ)
    elseif λ == 2
        return -log.(1 .- x)
    else
        @assert (x<0) && (λ!=2)
        return (expm1.(log.(1-x) .* (2-λ))) /(λ-2)
    end
end
function partialx(::YeoJohnson, x, λ)
    if x >= 0
        g = BoxCox()
        return partialx(g, x+1, λ)
    elseif λ == 2
        return 1 ./ (1 .- float(x))
    else
        @assert (x<0) && (λ!=2)
        (1-float(x)).^(1-λ)
    end
end


struct IdentityTransform <:NonlinearTransform end
(::IdentityTransform)(x, λ::Union{Array{T, 1}, T} where T<:Real) = (@assert (typeof(λ) <:Array{T} where T<:Real && size(λ, 2)==1) || (typeof(λ)<:Real); 
                                                            identity.(x))
partialx(::IdentityTransform, x, λ) = 1
partialxx(::IdentityTransform, x, λ)  = 0
partialλ(::IdentityTransform, x, λ) = 0
partialλλ(::IdentityTransform, x, λ) = 0
inverse(::IdentityTransform, x, λ) = identity.(x)
