"""
Computes linear polynomial basis
OUTPUT: 2D array
"""
function linear_polynomial_basis(x)#linear polynomial basis
    @assert typeof(x) <:Array{T, 1} where T  ||  typeof(x) <:Array{T, 2} where T
    if typeof(x)<:Array{T, 1} where T
        return hcat(1, reshape(x, 1, length(x)))
    else
        return hcat(ones(size(x, 1), 1), x)
    end
end
"""
Computes constant polynomial basis
OUTPUT: 2D Array
"""
function constant_basis(x)
    @assert typeof(x) <:Array{T, 1} where T  ||  typeof(x) <:Array{T, 2} where T
    if typeof(x) <:Array{T, 1} where T
        return reshape([1], 1, 1)
    else
        return reshape(ones(size(x, 1), 1), size(x, 1), 1)
    end
end

if false #test
a = constant_basis([1, 2, 3])
b = constant_basis(reshape([1; 2; 3], 3, 1))
c = linear_polynomial_basis([1 2; 3 4; 5 6])
d = linear_polynomial_basis([1;2;3])
end
