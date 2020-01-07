
module Functions

using Distributions


struct FunctionPrior{F<:Function,P<:Tuple{Vararg{ContinuousUnivariateDistribution}}}
    T::Type{F}
    p::P
end

function FunctionPrior(F::DataType, p::Vararg{ContinuousUnivariateDistribution})
    @assert !isabstracttype(F) "Abstract functions are not callable"
    @assert fieldcount(F) == length(p) "Number of priors does not match number of params"
    FunctionPrior{F, typeof(p)}(F, p)
end

abstract type Kernel <: Function end
abstract type Transform <: Function end


const kernels = [
    "SquaredExponential"
]

const transforms = [
    "BoxCox"
]

for f in kernels
    include(joinpath("kernels", "$(f).jl"))
end

for f in transforms
    include(joinpath("transforms", "$(f).jl"))
end

export
    FunctionPrior,
    # Kernels
    Kernel,
    SquaredExponential,
    # Transforms
    Transform,
    BoxCox

end # module
