include("../../validation/loocv.jl")

"""
Computes logdet hat(ΣθinvX) given logdet(ΣθinvX), where the hat indicates that
a new point has been incorporated.
"""
function extendlogdetΣθ(old_ΣθinvX, new_choleskyΣθ, new_X)
    lin_sys_loocv()
end