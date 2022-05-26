using GLMakie
using Makie.GeometryBasics

@views function PlotMakie(mesh, v, xmin, xmax, ymin, ymax, cmap = :viridis, min_v = minimum(v), max_v = maximum(v))
    ``` Create patch plots for either Quadrangles or Triangles
    ```
    # min_v, max_v = minimum(v), maximum(v)
    f = Figure()
    Axis(f[1, 1])
    p = [Polygon( Point2f0[ (mesh.xv[mesh.e2v[i,j]], mesh.yv[mesh.e2v[i,j]]) for j=1:mesh.nf_el] ) for i in 1:mesh.nel]
    poly!(p, color = v, colormap = cmap, strokewidth = 0, strokecolor = :black, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect_ratio = :equal, colorrange=(min_v,max_v)) #, xlims=(0.0,1.0) marche pas
    # GLMakie.scatter!(mesh.xf[mesh.bc.==3] ,mesh.yf[mesh.bc.==3] )
    Colorbar(f[1, 2], colormap = cmap, limits=[min_v max_v], flipaxis = true, size = 25, height = Relative(2/3) )
    display(f)
    save( string(@__DIR__, "/plot.png"), f)
    return 
end
