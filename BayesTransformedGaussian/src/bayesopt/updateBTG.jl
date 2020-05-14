"""
Atomic updateBTG function

Incorporate new data into btg object, namely triple (x_new, Fx_new, y_new)
"""
function updateBTG!(btg::btg, x_new::Array{T, 2} where T<:Real, Fx_new::Array{T, 2} where T<:Real, y_new:: Union{T, G} where T<:Real where G<:Array{F, 1} where F<:Real)
    @assert typeof(btg.trainingData) <: extensible_trainingData
    update!(btg.trainingData, x_new, Fx_new, y_new) #update training data once, and importantly, first

    for key in keys(btg.train_buffer_dict)
        update!(btg.train_buffer_dict[key], btg.trainingData) #update train_buffer_dict, so we don't recompute cholesky decompositions from scratch
    end

    for key in keys(btg.λbuffer_dict)
        update!(btg.λbuffer_dict[key], y_new, btg.g)
    end

    for key in keys(btg.θλbuffer_dict)
        update!(btg.θλbuffer_dict[key], btg.λbuffer_dict[key[2]], btg.train_buffer_dict[key[1]], btg.trainingData) #update θλbuffer
    end
    btg.n = btg.n + 1
end

