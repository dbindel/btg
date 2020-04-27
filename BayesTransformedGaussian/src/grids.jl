"""
Creates view object for grid
"""
function generate_view(grid, nt1::Int64, nt2::Int64, nl2::Int64, quadType::Array{String})
    if quadType == ["Gaussian", "Gaussian"] # 3d grid
        view_grid = @view grid[[1:nt2 for i = 1:nt1]..., 1:nl2]
    elseif (endswith(quadType[1], "Carlo") && endswith(quadType[2], "Carlo")) # 1d grid
        view_grid = @view grid[:]
    else # 2d grid
        view_grid = @view grid[:,:]
    end
    return view_grid
end

"""
Creates tensorgrids of functions computed with comp_tdist
INPUTS:
    - nt1: number of dimensions of theta
    - nt2: number of theta quadrature in each dimension
    - nl2: number of lambda quadrature in each dimension
    - derivatives: whether space should be allocated for derivatives
"""
function tgrids(nt1::Int64, nt2::Int64, nl2::Int64, quadType::Array{String}, weightsTensorGrid; derivatives = false)
    #preallocate some space to store dpdf, pdf, and cdf functions, as well as location parameters, for all (θ, λ) quadrature node combinations
    if quadType == ["Gaussian", "Gaussian"]
        tgridpdfderiv = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid)) #total num of length scales is num length scales of theta +1, because lambda is 1D
        tgridpdf = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid))
        tgridcdf = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid))
        tgridm = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid))
        tgridsigma_m = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid))
        tgridquantile = Array{Function, nt1+1}(undef, Base.size(weightsTensorGrid)) # store quantile of each T component
        if derivatives
            tgridcdf_augmented_deriv = similar(tgridcdf)
            tgridcdf_augmented_hess = similar(tgridcdf)
        else
            tgridcdf_augmented_deriv = nothing
            tgridcdf_augmented_hess = nothing
        end
    elseif endswith(quadType[1], "Carlo") && endswith(quadType[2], "Carlo")
        tgridpdfderiv = Array{Function, 1}(undef, Base.size(weightsTensorGrid)) #total num of length scales is num length scales of theta +1, because lambda is 1D
        tgridpdf = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
        tgridcdf = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
        tgridm = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
        tgridsigma_m = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
        tgridquantile = Array{Function, 1}(undef, Base.size(weightsTensorGrid))
        if derivatives
            tgridcdf_augmented_deriv = similar(tgridcdf)
            tgridcdf_augmented_hess = similar(tgridcdf)
        else
            tgridcdf_augmented_deriv = nothing
            tgridcdf_augmented_hess = nothing
        end
    else
        tgridpdfderiv = Array{Function, 2}(undef, Base.size(weightsTensorGrid)) #total num of length scales is num length scales of theta +1, because lambda is 1D
        tgridpdf = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
        tgridcdf = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
        tgridm = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
        tgridsigma_m = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
        tgridquantile = Array{Function, 2}(undef, Base.size(weightsTensorGrid))
        if derivatives
            tgridcdf_augmented_deriv = similar(tgridcdf)
            tgridcdf_augmented_hess = similar(tgridcdf)
        else
            tgridcdf_augmented_deriv = nothing
            tgridcdf_augmented_hess = nothing
        end
    end
    return (tgridpdfderiv, tgridpdf, tgridcdf, tgridm, tgridsigma_m, tgridquantile, tgridcdf_augmented_deriv, tgridcdf_augmented_hess)
end


