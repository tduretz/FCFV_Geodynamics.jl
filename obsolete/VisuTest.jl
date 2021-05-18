using PyPlot
pygui(true)

fig, ax = subplots()
xlim(0.0, 0.005)
ylim(0.0, 0.005)
grid()
        
circle1 = matplotlib[:patches][:Circle]((0.001, 0.001), radius = 0.001, facecolor="b")
circle2 = matplotlib[:patches][:Circle]((0.003, 0.003), radius = 0.001, facecolor="r")
p = matplotlib[:collections][:PatchCollection]([circle1, circle2])
ax[:add_collection](p)
show()