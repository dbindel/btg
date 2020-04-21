using LinearAlgebra
using DataFrames
using CSV
using Sobol
using FastGaussQuadrature
using Distributions
using DelimitedFiles
import Base: size



#This file contains various renditions of  which perform numerical
#quadrature given integration endpoints and a function handle
#More importantly, it provides a composite type which stores nodes and weight. 

"""
Data-type which stores quadrature nodes and weights. 
"""
struct nodesWeights
    nodes::Array{Float64, 2}
    weights::Array{Float64, 2}
    d::Int64 #number of dimensions/length scales
    num::Int64 #number of quadrature nodes
    #nodesWeights() = new([1 2; 3 4], [ 1 2 ; 3 4], 4, 4)
    function nodesWeights(ranges::Array{Float64, 2}, quadtype::String = "Gaussian"; num_pts = 12, num_MC = 200)
            d = size(ranges, 1)
            if quadtype == "Gaussian"
                N = Array{Float64, 2}(undef, d, num_pts)
                W = Array{Float64, 2}(undef, d, num_pts)
                nodes, weights = gausslegendre(num_pts)
                for i = 1:size(ranges, 1)
                    N[i, :], W[i, :] = affineTransform(nodes, weights, ranges[i, :])
                end
            elseif quadtype == "MonteCarlo"
                num_pts = num_MC 
                N = Array{Float64, 2}(undef, d, num_pts)
                W = ones(1, num_pts)
                for i = 1:size(ranges, 1)
                    N[i, :] = rand(Distributions.Uniform(ranges[i, 1], ranges[i, 2]), num_pts)
                end
            elseif quadtype == "QuasiMonteCarlo"
                num_pts = num_MC
                s = SobolSeq(ranges[:,1], ranges[:,2])
                N = hcat([next!(s) for i = 1:num_pts]...)
                W = ones(1, num_pts)
            elseif quadtype == "SparseGrid"
                level = d < 7 ? 8 : 4
                grids = readdlm("../quadrature/quadratureData/GQU/GQU_d$(d)_l$(level).asc", ',', Float64)
                num_pts = size(grids, 1)
                W = reshape(grids[:, end], 1, num_pts)
                # affine trans
                start = @view ranges[:, 1]; length = @views ranges[:, 2] .- start
                N = Diagonal(length) * (@view grids[:, 1:end-1])'  
                N = broadcast(+, N, start) 
                # W .*= reduce(*, length) -- for high dimen case, this reduce(*, length) could be large, so ignore the constant factor
            else 
                throw(ArgumentError("Quadrature type not supported. Choices are \"Gaussian\", \"MonteCarlo\", \"QuasiMonteCarlo\" and \"SparseGrid\""))
            end
            return new(N, W, d, num_pts)
    end    
end

"""
Get dimensions of nodes or weights matrix
"""
getNum(nw::nodesWeights) = nw.num
getDimension(nw::nodesWeights) = nw.d
size(nw::nodesWeights) = (nw.d, nw.num)
getNodeSequence(arr::Array{Float64, 2}, I) = (ret = [arr[i, j] for i =1:size(arr, 1) for j = I[i]];  length(ret) == 1 ? ret[1] : ret) #if ret val is array of length 1, then unboxes the array 
# getProd(arr::Array{Float64, 2}, I) = reduce(*, [arr[i, j] for i =1:size(arr, 1) for j = I[i]]) 
function getProd(w1::Array{Float64, 2}, w2::Array{Float64, 2}, I)
    n1 = size(w1, 1)
    n3 = size(w2, 1)
    nodes_tuple = vcat([w1[i, j] for i =1:n1 for j = I[i]], [w2[i, j] for i=1:n3 for j = I[i+n1]])
    return reduce(*, nodes_tuple) 
end

function getNumLengthScales(nw::nodesWeights)
    return nw.d
end

function getNodes(nw::nodesWeights)
    return nw.nodes
end

function getWeights(nw::nodesWeights)
    return nw.weights
end


"""
Apply affine transformation to nodes to integration over range r (linear change of variables)
Input nodes are assumed to be tailored to [-1, 1]. Transformation of nodes gives integration nodes
for new domain [r[1], r[2]]
"""
function affineTransform(nodes::O, weights::O, r::O) where O<:Array{Float64, 1}
    center = (r[2] + r[1]) /2
    length = (r[2] - r[1]) /2
    nodes = length .* nodes .+ center #shift and scale nodes
    weights = weights .* length #linarly scale weights
    return (nodes, weights)
end


"""
Uses the FastGaussQuadrature package to compute 
Gauss-Legendre quadrature nodes and weights for the domain [-1, 1]

INPUTS: n, the number of desired nodes
OUTPUTS: x and w, the nodes and weights
"""

#precomute various integration nodes/weights
dm0, wm0 = gausslegendre(24) 
dm, wm = gausslegendre(12) 
dm2, wm2 = gausslegendre(50)
dm3, wm3 = gausslegendre(100)

function getGaussQuadraturedata(n=12)
    dm, wm = gausslegendre(n)
    nodesWeights(dm, wm)
end

"""
INPUTS:
f is a function handle, a and b are integration endpoints (a<b), and n 
is the number of quadrature nodes (defaults to 10). Note that f can be
multi-output, in which case int1D integrates each entry individually. This way,
the function lends itself to being composed with Gauss-Turan integration.  
"""
function int1D(f, arr, num="1")
    if num=="1"
        x, w= dm, wm
    elseif num=="2"
        x, w= dm2, wm2
    elseif num=="0"
        x, w = dm0, wm0
    else 
        x, w = dm3, wm3
    end
    sample = f(arr[1])
    res = zeros(size(sample))
    for i = 1:length(x)
        expr = (arr[2]-arr[1])/2 * x[i] .+ (arr[1]+arr[2])/2
        #println(res)
        #println(f(expr) .* w[i])
        res = res .+ f(expr) .* w[i]
    end
    res = (arr[2]-arr[1])/2 .* res
end

function int1D_print(f, arr, num="0")
    if num=="1"
        x, w= dm, wm
    elseif num=="2"
        x, w= dm2, wm2
    elseif num=="0"
        x, w = dm0, wm0
    else 
        x, w = dm3, wm3
    end
    sample = f(arr[1])
    res = zeros(size(sample))
    for i = 1:length(x)
        expr = (arr[2]-arr[1])/2 * x[i] .+ (arr[1]+arr[2])/2
        println(res)
        println(f(expr) .* w[i])
        res = res .+ f(expr) .* w[i]
    end
    res = (arr[2]-arr[1])/2 .* res
end

"""
Integrates function over 2D domain using Gauss-Legendre Quadrature
arr[:, 1] are lower bounds, while arr[:, 2] are upper bounds     
"""
function int2D(f, arr)
    d = size(arr, 1) #number of dimensions
    int = 0.0
    x, w = dm, wm
    avg1 = (arr[1, 2] - arr[1, 1])/2
    avg2 = (arr[2, 2] - arr[2, 1])/2
    for i = 1:length(x)
        for j = 1:length(x)
           expr1 = avg1*x[i] + (arr[1, 2]+arr[1, 1])/2
           expr2 = avg2*x[j] + (arr[2, 2]+arr[2, 1])/2
           int = int + w[i]*w[j]*f(expr1, expr2)
        end
    end
    int = int*avg1*avg2
end

#turan nodes and weights
nodes12 = 
[-0.98644507017798, 
-0.913674511061696,
-0.781193008348874,
-0.597620339984639,
-0.374959012867354,
-0.127772153072766,
 0.127772153072766,
 0.374959012867354,
 0.597620339984638,
 0.781193008348874,
 0.913674511061695,
 0.986445070177978]

 weights12 = (
 a = [0.0410636498847265;
    0.103570747783452;
    0.159505143388351;
    0.205018290428583;
    0.237123354701031;
    0.253718813813857;
    0.253718813813857;
    0.237123354701032;
    0.205018290428581;
    0.159505143388351;
    0.10357074778345;
    0.0410636498847288;];

b = [0.000200708206472283;
    0.000473491788240051;
    0.000623732914397858;
    0.00061335624975073;
    0.000445103314005514;
    0.00016229083994099;
   -0.000162290839940938;
   -0.000445103314005562;
   -0.00061335624975073;
   -0.000623732914397801;
   -0.000473491788240044;
   -0.000200708206472284];

c = [1.75500948544866e-6;
   2.82815491676391e-5;
   0.000103308274307408;
   0.000219376538120014;
   0.000339418529291241;
   0.00041578672113985;
   0.000415786721139843;
   0.000339418529291247;
   0.000219376538120009;
   0.000103308274307411;
   2.8281549167637e-5 ;
   1.75500948544888e-6;];
    hcat(a, b, c)
 )


#data50 = DataFrame(CSV.File("quadratureData/nodes_weights_50.csv", header=0))
#nodes50 = convert(Array, data50[:,1]) #integration nodes for Gauss-Turan Quadrature
#weights50 = convert(Matrix, data50[:, 2:end]) #integration weights 

#data20 = DataFrame(CSV.File("quadratureData/nodes_weights_20.csv", header=0))
#nodes20  = convert(Array, data20[:,1])
#weights20 = convert(Array, data20[:,2:end])

#data12 = DataFrame(CSV.File("quadratureData/nodes_weights_12.csv", header=0))
#nodes12  = convert(Array, data12[:,1])
#weights12 = convert(Array, data12[:,2:end])

function getTuranQuadratureData()
    return nodesWeights(nodes12, weights12)
end

"""
Gauss-Turan integration with 2 derivatives
INPUTS:
arr contains enpoints of integration interval
"""
function Gauss_Turan(F, arr, nodes = nodes12, weights=weights12)
    a = arr[1]
    b = arr[2]
    nodes = (b-a)/2 .* nodes .+ (b+a)/2
    T = zeros(length(nodes), 3)
    for i = 1:length(nodes)
        (T[i, 1], T[i, 2], T[i, 3]) = F(nodes[i])
    end
    
    fn = T[:, 1]
    dfn = T[:, 2]
    d2fn = T[:, 3]
    return (fn'*weights[:, 1] + dfn'*weights[:, 2] + d2fn'*weights[:, 3])*(b-a)/2
end

"""
n-dimensional Gauss-Legendre integration
"""
function intnD(f, arr, n)

end
