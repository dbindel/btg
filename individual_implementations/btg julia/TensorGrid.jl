"""
Data-type which stores quadrature nodes and weights
"""
struct nodesWeights{T<:Array{Float64, 2}}
    nodes::T
    weights::T  
end

"""
    createTensorGrid(example, meshtheta, meshlambda, type)

Define a function ``f`` from ``R^k`` to ``Mat(n, n)``, such that ``f(z_0)_{ij} = p(z_0|z, θ_i, λ_j)``, 
where ``i`` and ``j`` range over the meshgrids over ``θ`` and ``λ``. Optional arg ``type`` is ""Gaussian""
by default. If ``type`` is "Turan", then use Gauss-Turan quadrature to integrate out ``0`` variable. 
"""
function getTensorGrid(example::setting{Array{Float64, 2}, Array{Float64, 1}}, meshθ::Array{Float64, 1}, meshλ::Array{Float64, 1}, nodesWeightsθ, nodesWeightsλ, type = "Gaussian")
    l1 = length(meshθ); l2 = length(meshλ); l3 = type == "Turan" ? 3 : 1
    function func_fixed(θ::Float64)
        return funcθ(θ, example, type)
    end
    if type=="Gaussian"
    elseif type == "Turan"
    else
        throw(ArgumentError("Quadrature type undefined. Please enter \"Gaussian\" or \"Turan\" for last arg."))
    end
    theta_param_list = Array{Union{θ_params{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}, θ_param_derivs{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}}}(undef, l1)
    for i=1:l1
        theta_param_list[i] = func_fixed(meshθ[i])
    end
    tgrid = Array{Any, 3}(undef, l1, l2, l3) #tensor grid
    for i = 1:l1
        for j = 1:l2 
            (f, df, d2f) = define_posterior(meshθ[i], meshλ[j], theta_param_list[i], example, type)
            for k = 1:l3
                tgrid[i, j, k] =2
            
            tgrid[i, j, 1] = f
            tgrid[i, j, 2] = df
            tgrid[i, j, 3] = d2f
        end
    end
end
    function evalTgrid(z0)
        res = Array{Float64, 3}(undef, l1, l2, l3)
        for i=1:l1
            for j = 1:l2
                for k =1:l3
                    res[i, j, k] = tgrid[i, j, k](z0)
                end
            end
        end 
        return res
    end
    return evalTgrid 
end