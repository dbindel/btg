include("../btg.jl")
include("constrained_guess.jl")
include("sample.jl")
include("../computation/finitedifference.jl")

using Plots
using Random
using Ipopt, JuMP
import JuMP.MathOptInterface
const MOI = MathOptInterface

include("optUCB.jl")
include("updateBTG.jl")




