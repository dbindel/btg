# full test of fast and naive validation 
using Dates
using ArgParse

include("../btg.jl")
include("../datasets/load_abalone.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "--fast"
        help = "use fast or not"
        action = :store_true
end
parsed_args = parse_args(ARGS, s)

if filesize("test_validation5_full.txt") == 0
    # write the setting headers
    io = open("test_validation5_full.txt", "w") 
    write(io, "Time         ;    ind      ;       θ    ;    λ   ;    x   ;   cov  ;  fast  ;   elapsedmin \n")
    close(io)
end

ind = 1:400
posx = 1:7
posc = 1:3
x = data[ind, posx] 
Fx = data[ind, posc] 
y = float(target[ind])

#prediction data
pind = 1:1
Fx0 = reshape(data[pind, posc], 1, length(posc))
x0 = reshape(data[pind, posx], 1, length(posx)) 

#parameter setting
rangeθ = [10.0 1000]
rangeλ = [0. 3.] #we will always used 1 range scale for lambda


trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
btg1 = btg(trainingData1, rangeθ, rangeλ; quadtype = ["SparseGrid", "Gaussian"])
(pdf, cdf, dpdf) = solve(btg1); #initialize training_buffer_dicts, solve once so can use fast techiques to extrapolate submatrix determinants, etc.


"""
leave-one-out-training-data
"""
function lootd(td::AbstractTrainingData, i::Int64)
    x = getPosition(td)
    Fx = getCovariates(td)
    z = getLabel(td)
    x_minus_i = x[[1:i-1;i+1:end], :]
    Fx_minus_i = Fx[[1:i-1;i+1:end], :]
    z_minus_i = z[[1:i-1;i+1:end]]
    x_i = x[i:i, :]
    Fx_i = Fx[i:i, :]
    z_i = z[i:i, :]
    return trainingData(x_minus_i, Fx_minus_i, z_minus_i), x_i, Fx_i, z_i
end

nrow = 6; ncol = 8
y1_temp = Array{Float64, 1}(undef, 100)
y2_temp = Array{Float64, 1}(undef, 100)
y1_set = Array{Float64, 2}(undef, 100, ncol*nrow)
y2_set = Array{Float64, 2}(undef, 100, ncol*nrow)
xgrid = range(.01, stop=1.2, length=100)
before = Dates.now()
for j = 1:n
    mod(j, 10) == 0 ? (@info j) : nothing
    if parsed_args["fast"]
        (pdf, cdf, dpdf) = solve(btg1, validate = j);   
        b1 = y0 -> pdf(x0, Fx0, y0);
        c1 = y0 -> cdf(x0, Fx0, y0);
    else
        (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, j)
        btg2 = btg(trainingdata_minus_i, rangeθ, rangeλ; quadtype = ["SparseGrid", "Gaussian"])
        (pdf_minus_i, cdf_minus_i, dpdf_minus_i) = solve(btg2)
        b1 = y -> pdf_minus_i(x_i, Fx_i, y) 
        c1 = y -> cdf_minus_i(x_i, Fx_i, y) 
    end
    y1_temp = b1.(xgrid)
    y2_temp = c1.(xgrid)
    if j <= ncol*nrow
        y1_set[:, j] = y1_temp
        y2_set[:, j] = y2_temp
    end
end
after = Dates.now()
elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)

io = open("test_validation5_full.txt", "a") 
write(io, "$(Dates.now())  ;    $ind      ;       $rangeθ    ;    $rangeλ   ;    $posx   ;   $posc  ;  $(parsed_args["fast"])  ;   $elapsedmin \n")
close(io)

## Plot
PyPlot.close("all") #close existing windows
plt, axs = PyPlot.subplots(nrow, ncol)
PyPlot.suptitle("Cross Validation $(parsed_args["fast"])", fontsize=10)
for j = 1:nrow*ncol
    ind1 = Int64(ceil(j/ncol))
    ind2 = Int64(j - ncol*(floor((j-.1)/ncol)))
    axs[ind1, ind2].plot(xgrid, y1_set[:, j], color = "red", linewidth = 1.0, linestyle = "-")
    axs[ind1, ind2].plot(xgrid, y2_set[:, j], color = "orange", linewidth = 1.0, linestyle = "-")
    axs[ind1, ind2].axvline(x = getLabel(btg1.trainingData)[j])
end
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs
    ax.label_outer()
end
PyPlot.savefig("figure/test_v5_ind_$(ind[1])_$(ind[end])_rθ_$(Int(rangeθ[1]))_$(Int(rangeθ[2]))_rλ_$(Int(rangeλ[1]))_$(Int(rangeλ[2]))_$(parsed_args["fast"]).pdf")