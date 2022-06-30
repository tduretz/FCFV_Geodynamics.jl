using GLMakie
using Makie.GeometryBasics

# ENV["MPLBACKEND"]="Qt5Agg"
# using PyPlot
# pygui(true)

@views function PlotMakie(mesh, v, xmin, xmax, ymin, ymax; cmap = :viridis, min_v = minimum(v), max_v = maximum(v), writefig=false)
    ``` Create patch plots for either Quadrangles or Triangles
    ```
    # min_v, max_v = minimum(v), maximum(v)
    f = Figure()
    ar = (maximum(mesh.xv) - minimum(mesh.xv)) / (maximum(mesh.yv) - minimum(mesh.yv))
    Axis(f[1, 1], aspect = ar)
    p = [Polygon( Point2f0[ (mesh.xv[mesh.e2v[i,j]], mesh.yv[mesh.e2v[i,j]]) for j=1:mesh.nf_el] ) for i in 1:mesh.nel]
    poly!(p, color = v, colormap = cmap, strokewidth = 0, strokecolor = :black, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect_ratio=:image, colorrange=(min_v,max_v)) #, xlims=(0.0,1.0) marche pas
    # GLMakie.scatter!(mesh.xf[mesh.bc.==3] ,mesh.yf[mesh.bc.==3] )
    Colorbar(f[1, 2], colormap = cmap, limits=(min_v, max_v), flipaxis = true, size = 25, height = Relative(2/3) )
    resize_to_layout!(f)
    display(f)
    if writefig==true 
        save( string(@__DIR__, "/plot.png"), f)
    end
    return 
end

# function PlotPyPlot(mesh, v, xmin, xmax, ymin, ymax; cmap = :viridis, min_v = minimum(v), max_v = maximum(v), writefig=false)

#     # Prepare mesh for visalisation
#     p = fill(Float64[], mesh.nv)
#     for i=1:mesh.nv
#         p[i] = [mesh.xv[i], mesh.yv[i]]
#     end
#     t = fill(Int64[], mesh.nel)
#     for i=1:mesh.nel
#         t[i] = [mesh.e2v[i,1], mesh.e2v[i,2], mesh.e2v[i,3]]
#     end

#     # Plot triangular mesh with nodes `p` and triangles `t`
#     clf()
#     tris = convert(Array{Int64}, hcat(t...)')
#     display(tripcolor(first.(p), last.(p), tris .- 1, v,
#               cmap="viridis", edgecolors="none", linewidth=0) )
#     axis("equal")
#     ylim([0, 1])
#     xlim([0, 1])
#     title("Low res.")
#     xlabel("x")
#     ylabel("y")
#     colorbar()
#     show()
#     return 
# end


