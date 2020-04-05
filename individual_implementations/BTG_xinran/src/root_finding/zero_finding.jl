# find the interval for integral for a pdf

function zero_finding(f, kmax, tol, b0)
    # step 0:
    # test if b0 is a good guess
    while f(b0) < tol
        b0 = b0/2
    end

    b = b0
    k = 0
    iter = 0
    intvl = [b, 2*b]


    # step 1:
        # find an interval [a, b] such that f(a) > tol, f(b) < tol 
    while (f(intvl[1]) > tol && f(intvl[2]) < tol) == false
        intvl[1] = intvl[2]
        intvl[2] *= 2
        # print("Current interval: [$(intvl[1]), $(intvl[2])] \n")
        # print("Function values: [$(f(intvl[1])), $(f(intvl[2]))] \n")
        iter += 1
    end

    # step 2
        # shrink such interval using bisection
    while k < kmax
        midpt = (intvl[1] + intvl[2])/2
        midvalue = f(midpt)
        if midvalue < tol
            intvl[2] = midpt
            # print("Current interval: [$(intvl[1]), $(intvl[2])] \n")
            # print("Function values: [$(f(intvl[1])), $(f(intvl[2]))] \n")
        else
            intvl[1] = midpt
            # print("Current interval: [$(intvl[1]), $(intvl[2])] \n")
            # print("Function values: [$(f(intvl[1])), $(f(intvl[2]))] \n")
        end
        k += 1
    end
    ttl_iter = iter + k
    return intvl[2], ttl_iter
end
