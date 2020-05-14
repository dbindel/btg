
function testdata()
    X = [0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    y = [0.5, 1.0, 1.5, 2.0]
    f = Constant()
    Fx = covariate(f, X)
    g = Identity()
    k = Gaussian()
    return X, y, Fx, f, k, g
end
