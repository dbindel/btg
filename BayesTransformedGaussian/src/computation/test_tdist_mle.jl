include("finitedifference.jl")
include("tdist_mle.jl")
using Plots

function flatten_reshape(A::Array{T} where T<:Array{G} where G)
    B = reshape(A, length(A), 1)
    C = zeros(length(B), length(B[1, 1]))
    for i = 1:size(C, 1)
        for j = 1:size(C, 2)
            C[i, j] = B[i][j]
        end
    end
    return C
end

btg1 = btg(trainingData([1 2; 3 4; 5 6.0], reshape([1.0 ; 1.0; 1.0], 3, 1), [2, 3, 4.0]), [1 10.0; 2 12.0], [-0.5 0.5]);

f = x ->  (w = Σθ(btg1, x); reshape(w, length(w), 1));
g = x -> (w = partial_Σθ_theta(btg1, x); flatten_reshape(w));

(r1, r2, plt1, pol1) = checkDerivative(f, g, [1.1, 2.1], nothing);