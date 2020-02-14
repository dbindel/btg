# given an interval, return quadrature points and weights

using LinearAlgebra

function quadgrid(xmin, xmax, n)
    dist = (xmax - xmin)/2
    center = (xmax + xmin)/2
    if n == 1
        w = 2
        x = 0
    elseif n == 2
        w = [1, 1]
        x =  1/sqrt(3) * [1, -1]
    elseif n == 3
        w = [5/9, 8/9, 5/9]
        x = -sqrt(3/5) * [-1, 0, 1]
    elseif n == 4
        w = [(18+sqrt(30))/36, (18+sqrt(30))/36,
             (18-sqrt(30))/36, (18-sqrt(30))/36]
        x = [sqrt(3/7 - 2/7*sqrt(6/5)), -sqrt(3/7 - 2/7*sqrt(6/5)), 
             sqrt(3/7 + 2/7*sqrt(6/5)), -sqrt(3/7 + 2/7*sqrt(6/5))]
    elseif n == 5
        w = [128/225, (322+13*sqrt(70))/900, (322+13*sqrt(70))/900,
                      (322-13*sqrt(70))/900, (322-13*sqrt(70))/900]
        x = [0, sqrt(5-2*sqrt(10/7))/3, -sqrt(5-2*sqrt(10/7))/3,
                sqrt(5+2*sqrt(10/7))/3, -sqrt(5+2*sqrt(10/7))/3]
    end

    w_new = dist .* w
    x_new = dist .* x .+ center
    # fx = f.(x_new)
    # int = dot(w_new, fx)

    return x_new, w_new
end
