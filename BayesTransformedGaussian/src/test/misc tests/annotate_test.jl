

using PyPlot
if false 
w = [3, 4]
fig, axs = PyPlot.subplots(1, 1)
annotate(
L"$\theta: $" * "$w" * ".\n" *
L"$\lambda: $" * "$range" * ".\n"*
"quad: " * "$quadType",
 xy=[.1;.2],
    fontsize=80.0
)
end
rangeθ = [2, 3]
plt, axs = PyPlot.subplots(2, 2)
axs[2, 2].annotate(L"$\theta: $" * "$rangeθ" * ".\n",  xy=[.1;.2],
fontsize=80.0)

    #xycoords="axes fraction",
	# xytext=[-10,10],
	# textcoords="offset points",
    #
    #
    #ha="center",
    #va="bottom"
    
    #	xy=[.4;.5],
    