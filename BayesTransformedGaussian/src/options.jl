abstract type BtgOptions end


"""
Options in btg model
TODOs: 
    should have a default/suggested rangeθ and rangeλ for users
    could add MLE/bayesian options here
"""
mutable struct Options<:BtgOptions
    # model options
    transform_type::String 
    kernel_type::String 
    quadrature_type::Dict{String,String} # specify the type of discretization for different BTG model parameters
    quadrature_size::Dict{String,Int} # specify the grid size for different types of discretization
    # model parameter options
    parameter_range::Dict{String,Array{T, 2}} where T<:Real
    parameter_prior::Dict{String,priorType}  
    parameter_dimθ::Int # dimension of theta, 1 or d, i.e. single lengthscale or dimension of input

    # prediction options
    confidence_level::T where T<:Real

    function Options(rangeθ::Array{T, 2}, rangeλ::Array{T, 2}, trainingData::AbstractTrainingData, parameter_dimθ::Int = 1) where T<:Real
        @assert size(rangeλ) == (1, 2) "Should provide 1-dimensional and two-end range of θ"
        @assert size(rangeθ)[2] == 2 "Should provide two-end range of θ"
        if parameter_dimθ > 1
            d = getDimension(trainingData)
            @assert d == parameter_dimθ "Multi-dimensional parameter θ should match dimension of input data"
            # @assert d == parameter_dimθ
        end
        if parameter_dimθ > 1 && size(rangeθ)[1] == 1
            rangeθs = repeat(rangeθ, parameter_dimθ, 1) # adjust rangeθs to match parameter_dimθ
        elseif parameter_dimθ == 1 && size(rangeθ)[1] > 1
            @warn "Given multiple lengthscale range, BTG adjusting parameter_dimθ to match."
            parameter_dimθ = size(rangeθ)[1] 
            rangeθs = rangeθ
        else 
            rangeθs = rangeθ
        end
        transform_type = "BoxCox" # else: "IdentityTransform", "ShiftedBoxCox"
        kernel_type = "Gaussian" 
        quadrature_type = Dict("θ" => "Gaussian", "λ" => "Gaussian") # recommended combo: ["SparseGrid", "GaussianQuadrature"], ["SparseCarlo", "SparseCarlo"], ["QuasiMonteCarlo", "QuasiMonteCarlo"]
        quadrature_size = Dict("Gaussian" => 12, "MonteCarlo" => 400) # for "SparseCarlo" and "SparseGrid", depends on input dimension.
        parameter_range = Dict("θ" => rangeθs, "λ" => rangeλ)
        parameter_prior = Dict("θ" => inverseUniform(rangeθ), "λ" => Uniform(rangeλ))
        confidence_level = .95
        new(transform_type, kernel_type, quadrature_type, quadrature_size, parameter_range, parameter_prior, parameter_dimθ, confidence_level)
    end
end

function print(O::BtgOptions)
    println("\n\n=============== BTG OPTIONS =============== ")
    println("   transform type:       $(O.transform_type)")
    println("   Kernel type:          $(O.kernel_type)")
    println("   quadrature type:      $(O.quadrature_type)")
    println("   quadrature size:      $(O.quadrature_size)")
    println("   Confidence level:     $(O.confidence_level)")
    println("   Parameter range:      ")
    for key in keys(O.parameter_range)
        println("                         $(key): $(O.parameter_range[key])")
    end
    println("   Parameter prior:      ")
    for key in keys(O.parameter_prior)
        println("                         $(key): $(O.parameter_prior[key])")
    end
end


