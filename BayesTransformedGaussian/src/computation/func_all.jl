

include("functest.jl")
include("funcfile.jl")



function funifun(f, x)
 return f(x)
end

funifun(anonfun, 5)