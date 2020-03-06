using LinearAlgebra
using Distributions
using SpecialFunctions
using PDMats
using CSV
include("kernel.jl")
include("integration.jl")
include("btgDerivatives.jl")
include("examples.jl")

if false
    (main, dmain, d2main) = partial_theta(10, 2, getExample(1, 1000, 3, 3, 10))

    f = x -> main(x)[1]
    df = x -> dmain(x)[1]
    d2f = x -> d2main(x)[1]

    @time begin
    println(main([.4]))
    end
    @time begin
    println(dmain([.4]))
    end
    @time begin
    println(d2main([.4]))
    end

    #x = .01:.01:2
    #y = f.(x)
    #dy = df.(x)
    #d2y = d2f.(x)
    #plt1 = plot(x, y, title = "f")
    #plt2 = plot(x, dy, title = "df")
    #plt3 = plot(x, d2y, title = "d2f")
    #display(plot(plt1, plt2, plt3))
end
if true
    h = 50:50:500
    times = zeros(3, length(h))
    for i = h
        s1 = rand(10*i, 3)
        Z = K(s1, s1, 10)
        times[1, Int(i/50)] = @elapsed cz = cholesky(Z)
        times[2, Int(i/50)] = @elapsed Z\rand(size(Z, 1), 1)
        times[3, Int(i/50)] = @elapsed cz\rand(size(Z, 1), 1)
    end
    plot(repeat(h, 1,3), times')
end


