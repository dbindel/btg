"""
    This is a primitive attempt of translation for btg013
"""

using BtgError

function betacf(a::Float64, b::Float64, x::Float64)
    """
    Refer to the function betacf()

    a, b, x are floats.
    Returns the result of (incomplete) beta function
    Perhaps a helper function useful for t cdf?
    """
    err = 3e-7
    itermax = 100
    az = am = bm = 1.0
    bz = 1.0 - (a + b) * x / (a + 1.0)
    for m = 1: itermax
        d = m * (b - m) * x / ((a - 1 + 2 * m)*(a + 2 * m))
        ap = az + d * am;
        bp = bz + d * bm;
        d = - (a + m) * ((a + b) + m) * x / (((a - 1.0) + 2 * m) * (a + 2 * m))
        app = ap + d * az
        bpp = bp + d * bz
        aold = az
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1
        if abs(az - aold) < err * abs(az)
            return az
        end
    end
    BtgError.set_msg("Error: Could not compute Student t cdf")
end

# powi(x, i) not necessary. This is just x^i

function Student(t::Float64, nu::Int64)
    """
    Refer to Student() function

    Returns the value of t_cdf with nu degrees of freedom
    Requires: t is float, nu is int
    """
    x = nu / (nu + t * t)
    a = nu / 2
    b = 0.5

    if (x == 0 || x == 1.0)
        bt = 0
    else
        bt = exp(lgamma((nu+1)/2.0)-lgamma(nu/2.0))*sqrt((x^nu)*(1-x)/pi);
    end
    if (x < (nu+2.0)/(nu+5.0))
        bix = bt * betacf(a,b,x)/a;
    else
        bix = 1.0-bt * betacf(b, a, 1.0-x)/b;
    end
    if (t >= 0)
        return 1 - bix/2;
    else
        return bix/2;
end

# uniform number generator is not necessary.
# Use rand(MersenneTwister(), min:max)
