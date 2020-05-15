using Dates
using PyPlot
using Plots
#using ProfileView

include("../../btg.jl")
include("../../datasets/load_abalone.jl")

#ind = 350:370
#ind = 3300:3359
ind = 3300:3320
#posx = 1:3 #
#posx = [1;4]
posx = 1:7
#posc = 1:1
posc = 1:1
x = data[ind, posx] 
#choose a subset of variables to be regressors for the mean
Fx = data[ind, posc] 
y = float(target[ind])/29
pind = 10:10 #prediction index
#pind = i:i
trainingData1 = trainingData(x, Fx, y) #training data used for testing various functions
#prediction data
d = getDimension(trainingData1); n = getNumPts(trainingData1); p = getCovDimension(trainingData1)
Fx0 = reshape(data[pind, posc], 1, length(posc))
x0 = reshape(data[pind, posx], 1, length(posx)) 
rangeθ = [100.0 200]
#rangeθ = [500.0 1500]
rangeλ = [-1 1.0] #we will always used 1 range scale for lambda
btg1 = btg(trainingData1, rangeθ, rangeλ; quadtype = ["Gaussian", "Gaussian"])
#θ1 = btg1.nodesWeightsθ.nodes[1, 6] #pick some theta value which doubles as quadrature node
#λ1 = btg1.nodesWeightsλ.nodes[3]
##################################################################################################
PyPlot.close("all") #close existing windows

#(pdf, cdf, dpdf) = solve(btg1); #initialize training_buffer_dicts, solve once so can use fast techiques to extrapolate submatrix determinants, etc.
#pdf(x0, Fx0, .4);
#(pdf, cdf, dpdf) = solve(btg1; validate = i) #do LOOCV this time around
#a1 = y-> pdf(x0, Fx0, y)


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
"""
Run
"""
function test_LOOCV(btg1::btg, m::Int64, n::Int64; fast)
    fast = fast
    m = m; n=n
    #plt, axs = PyPlot.subplots(m, n,figsize=[20,11])
    plt, axs = PyPlot.subplots(m, n)
    #figure(1)
    before = Dates.now()
    println("LOOCV type is fast?: $fast")
    for j = 1:m*n
        ind1 = Int64(ceil(j/n))
        ind2 = Int64(j - n*(floor((j-.1)/n)))
        println("iteration $j")
        if j <m*n
            if fast
                (pdf, cdf, dpdf) = solve(btg1, validate = j);   
                b1 = y0 -> pdf(x0, Fx0, y0);
                c1 = y0 -> cdf(x0, Fx0, y0);
                (x, y) = plt_data(b1, .01, 1.2, 100)
                (xc, yc) = plt_data(c1, .01, 1.2, 100)
            else
                (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, j);
                #btg2 = btg(trainingdata_minus_i, rangeθ, rangeλ; quadtype = ["MonteCarlo", "MonteCarlo"])
                btg2 = btg(trainingdata_minus_i, rangeθ, rangeλ; quadtype = ["Gaussian", "Gaussian"]);
                (pdf_minus_i, cdf_minus_i, dpdf_minus_i) = solve(btg2);
                b1 = y -> pdf_minus_i(x_i, Fx_i, y);
                c1 = y -> cdf_minus_i(x_i, Fx_i, y);
                (x, y) = plt_data(b1, .01, 1.2, 100);
                (xc, yc) = plt_data(c1, .01, 1.2, 100);
            end
            axs[ind1, ind2].plot(x, y, color = "red", linewidth = 2.0, linestyle = ":")
            axs[ind1, ind2].plot(xc, yc, color = "orange", linewidth = 2.0, linestyle = ":")
            #axs[ind1, ind2].plot(x1, y1, color = "blue", linewidth = 1.0, linestyle = "--")
            axs[ind1, ind2].axvline(x =  getLabel(btg1.trainingData)[j])
        else #do annotation in last box
            #println("annotating...")
            after = Dates.now()
            elapsedmin = round(((after - before) / Millisecond(1000))/60, digits=5)
            qtype =  btg1.quadType
            axs[ind1, ind2].annotate(
            L"$\theta: $" * "$rangeθ" * "\n" *
            L"$\lambda: $" * "$rangeλ" * "\n" *
            "$qtype \n" * 
            "ind: " * "$ind \n" * 
            "x: " * "$posx " *", cov: " * "$posc\n" * 
            "time: " * "$elapsedmin", 
            xy=[.03;.03],
            fontsize = 5.0)
            # *
            #L"$\lambda: $" * "$rangeλ" * ".\n" *
            #"quad: " * "$btg1.quadType",
        end
    end
    #for ax in axs
    #    ax.set(xlabel="x-label", ylabel="y-label")
    #end
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs
        ax.label_outer()
    end
end

if true
    test_LOOCV(btg1, 4, 5; fast = false);
end

###
### reserved REPL testing block
###

if true
i=3
#function isolate(i)
    (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, i);
    btg2 = btg(trainingdata_minus_i, rangeθ, rangeλ; quadtype = ["Gaussian", "Gaussian"]);
    #(pdf, cdf, dpdf, augmented_cdf_deriv, augmented_cdf_hess, quantInfo, tgridpdf, tgridcdf, tgridm, tgridsigma_m, weightsTensorGrid)  = solve(btg2);
    (pdf_minus_i, cdf_minus_i, dpdf_minus_i, _, _, _, tgridpdf, tgridcdf, tgridm, tgridsigma, weightsTensorGrid) = solve(btg2);

    function func_grid_eval(funcgrid, x_i, Fx_i, y)
        g = similar(weightsTensorGrid)
        R = CartesianIndices(weightsTensorGrid)
        for I in R
            g[I] = funcgrid[I](x_i, Fx_i, y)
        end
        return g
    end
    b1 = y -> pdf_minus_i(x_i, Fx_i, y);
    c1 = y -> cdf_minus_i(x_i, Fx_i, y);
    d1 = y->  func_grid_eval(tgridpdf, x_i, Fx_i, y);
    e1 = y-> func_grid_eval(tgridcdf, x_i, Fx_i, y);
    f1 = () -> func_grid_eval(tgridm, x_i, Fx_i, y);
    g1 = () -> func_grid_eval(tgridsigma, x_i, Fx_i, y);
    (x, y) = plt_data(b1, .01, 1.2, 100);
    (xc, yc) = plt_data(c1, .01, 1.2, 100);
    #return (b1, c1, d1, e1, f1, g1, weightsTensorGrid)
#end

if false
    (trainingdata_minus_i, x_i, Fx_i, z_i) = lootd(trainingData1, 3);
    (b1, c1, d1, e1, f1, g1, w) = isolate(3);
end


#manual computation for checking

#btg2.θλbuffer.θ
#btg2.θλbuffer.λ
#btg2.θλbuffer.Σθ_inv_y

# testing data
x0_cur = btg2.testingData.x0;
Fx0_cur = btg2.testingData.Fx0;
#fix theta and lambda - run thru entire computation to get m, and see if it's negative...
theta_cur = btg2.nodesWeightsθ.nodes[1];
lambda_cur = btg2.nodesWeightsλ.nodes[1];
#computation by hand
Σθ_cur = correlation(Gaussian(), theta_cur, getPosition(trainingdata_minus_i); jitter = 1e-8);
Σθ_inv_X_cur = Σθ_cur\trainingdata_minus_i.Fx;
Bθ_cur = cross_correlation(Gaussian(), theta_cur, x0_cur, getPosition(trainingdata_minus_i));
Fx_cur = getCovariates(trainingdata_minus_i);
glz_cur = copy((getLabel(trainingdata_minus_i)));
for i = 1:length(glz_cur)
    glz_cur[i] = btg2.g(glz_cur[i], lambda_cur);
end
beta_hat_cur = (Fx_cur' * (Σθ_cur\Fx_cur))\(Fx_cur'*(Σθ_cur\glz_cur));
H_theta_cur = Fx_i - Bθ_cur*(Σθ_cur\Fx_cur);
m_cur = Bθ_cur*(Σθ_cur\glz_cur) + H_theta_cur*beta_hat_cur

#how sigma theta is computed in code
m=20;
Σθ = Array{Float64}(undef, 1000, 1000);
Σθ[1:m, 1:m] = correlation(Gaussian(), theta_cur, getPosition(trainingdata_minus_i)[1:m, :]; jitter = 1e-8);
choleskyΣθ = incremental_cholesky!(Σθ, m);

#stuff computed using btg code
Σθ_inv_X = btg2.train_buffer_dict[theta_cur].Σθ_inv_X;
cholΣθ = btg2.train_buffer_dict[theta_cur].choleskyΣθ;
Bθ = btg2.test_buffer_dict[theta_cur].Bθ;
gλz = btg2.λbuffer_dict[lambda_cur].gλz;
βhat = btg2.θλbuffer_dict[(theta_cur, lambda_cur)].βhat;
Hθ =  btg2.test_buffer_dict[theta_cur].Hθ;


id = Diagonal(ones(trainingData1.n-1));
rhs = rand(trainingData1.n-1, 1);

@assert norm(trainingdata_minus_i.x - btg2.trainingData.x) <0.0001 #training data matches up
@info "Σθ_inv_X_cur - Σθ_inv_X", norm(Σθ_inv_X_cur - Σθ_inv_X);
@info "rand Σθ_cur norm diff", norm(Σθ_cur\rhs - cholΣθ\rhs);
@info "Bθ_cur - Bθ", norm(Bθ - Bθ_cur);
#@info "rand Σθ_cur norm diff", norm(cholΣθ\rhs - choleskyΣθ\rhs);
@info "sigma theta inv B theta", norm(Σθ_cur\Bθ' - choleskyΣθ\Bθ');
@info "sigma theta inv B same matrix", ( ccc = cholesky(Σθ_cur);  norm(Σθ_cur\Bθ' - ccc\Bθ'));
@info "glz nrom diff", norm(glz_cur - gλz);
@info "betahat norm diff", norm(βhat - beta_hat_cur);
#@info btg2.nodesWeightsθ.nodes
#@info btg2.nodesWeightsλ.nodes
#@info btg2.train_buffer_dict[btg2.nodesWeightsθ.nodes[1]].choleskyΣθ
end


