"""
Samples f : R^d -> R at num points in box specified by lx and ux

INPUTS:
f: function being sampled
lx: length d array of lower bounds
ux: length d array of upper bounds

OUTPUTS:
res: length num array of sampled values
loc: num x d matrix of sample locations
"""
function sample(f, lx, ux; num = 10)
    @assert length(lx) == length(ux)
    d = length(lx)
    loc = rand(num, d)
    for j = 1:num
        for i = 1:d 
            loc[j, i] = loc[j, i]*(ux[i] - lx[i]) + lx[i]
        end
    end 
    res = zeros(num)
    for j  = 1:num
        res[j] = f(loc[j, :])
    end
    return res, loc
end


