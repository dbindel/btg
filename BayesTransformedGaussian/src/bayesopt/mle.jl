

"""
find upper and lower bounds on range_theta based on MLE estimates for extremal range_lambda values
"""
function mle(btg, glz, range_lambda, cholesky)
    N = 

    g = btg.g
    lmin = range_lambda[1]
    lmax = range_lambda[2]

    glzs1 = copy(glz)
    for i = 1:length(glz)
        glzs1[i] = g.(glz[i], lmin)
    end
    glzs2 = copy(glz)
    for i = 1:length(glz)
        glzs2[i] = g.(glz[i], lmax)
    end

    function choltheta

    L1 = 0.5*logdet(cholesky) + 0.5 * glzs1



    
end