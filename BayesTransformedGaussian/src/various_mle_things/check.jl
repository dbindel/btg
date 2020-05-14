function check_derivative(f, g, h, θ)
       a = (f.(θ .+ h) .- f(θ)) ./ h
       b = g(θ)
       return a,b
end

