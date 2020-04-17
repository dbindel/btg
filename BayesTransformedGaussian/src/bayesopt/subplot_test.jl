using PyPlot

y = sin.(collect(.1:.05:1))
x = collect(.1:.05:1)  
fig, axs = plt.subplots(2, 2)
axs[1, 1].plot(x, y)
axs[1, 1].set_title("Axis [0,0]")
axs[1, 2].plot(x, y, "tab:orange")
axs[1, 2].set_title("Axis [0,1]")
axs[2, 1].plot(x, -y, "tab:green")
axs[2, 1].set_title("Axis [1,0]")
axs[2, 2].plot(x, -y, "tab:red")
axs[2, 2].set_title("Axis [1,1]")

for ax in axs
    ax.set(xlabel="x-label", ylabel="y-label")
end
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs
    ax.label_outer()
end