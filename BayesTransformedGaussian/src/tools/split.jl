using Random

"""
Generate indices of train-test split given training indices
and percent training data
"""
function sample(inds, percent)
    N = length(inds)
    num = (Int64)(round(percent * N))
    #train = sort(rand(1:490, (num,)))
    ind_train = sort(randperm(N)[1:num])
    ind_test = copy(inds)
    deleteat!(ind_test, ind_train)
    #work with indices up until this point
    x_train = xs[ind_train]
    x_test = xs[ind_test]
    return ind_train, x_train, ind_test, x_test 
end

