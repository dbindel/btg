include("integration.jl")

function define_posterior(θ::Float64, λ::Float64, theta_params::θ_params{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}, example::setting{Array{Float64, 2}}, type = "Gaussian")
    #time = @elapsed begin
    pθ = x->1 
    dpθ = x->0
    dpθ2 = x->0
    pλ = x->1
    if type == "Gaussian"
    
    end
    (main, dmain, d2main) = partial_theta(float(θ), float(λ), example, theta_params, type)
    (main1, dmain1, d2main1) = posterior_theta(float(θ), float(λ), pθ, dpθ, dpθ2, pλ, example, theta_params, type)
    f = z0 -> (main(z0)*main1); df = z0 -> (main(z0)*dmain1 .+ dmain(z0)*main1); d2f = z0 -> (d2main(z0)*main1 .+ main(z0)*d2main1 .+ 2*dmain(z0)*dmain1)
    #end
    #println("define_fs time: %s\n", time)
    obj = (f, df, d2f)  #named tuple
    return (f, df, d2f)
end

"""
    createTensorGrid(example, meshtheta, meshlambda, type)

Define a function ``f`` from ``R^k`` to ``Mat(n, n)``, such that ``f(z_0)_{ij} = p(z_0|z, θ_i, λ_j)``, 
where ``i`` and ``j`` range over the meshgrids over ``θ`` and ``λ``. Optional arg ``type`` is ""Gaussian""
by default. If ``type`` is "Turan", then use Gauss-Turan quadrature to integrate out ``0`` variable. 
"""
function getTensorGrid(setting::setting{T, Array{Float64, 1}}, nodesWeightsθ::nodesWeights{T}, nodesWeightsλ::nodesWeights{T}) where T<:Array{Float64, 2}
    meshθ = nodesWeightsθ.nodes
    meshλ = nodesWeightsλ.nodes
    l1 = length(meshθ); l2 = length(meshλ); l3 = type == "Turan" ? 3 : 1
    function func_fixed(θ::Float64)
        return funcθ(θ, setting, type)
    end
    theta_param_list = Array{Union{θ_params{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}, θ_param_derivs{Array{Float64, 2}, Cholesky{Float64,Array{Float64, 2}}}}}(undef, l1)
    for i=1:l1
        theta_param_list[i] = func_fixed(meshθ[i])
    end
    tgrid = Array{Any, 3}(undef, l1, l2, l3) #tensor grid
    for i = 1:l1
        for j = 1:l2 
            funcs = define_posterior(meshθ[i], meshλ[j], theta_param_list[i], setting, type)
            for k = 1:l3
                tgrid[i, j, k] = funcs[k]
            
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


