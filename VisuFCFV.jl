using GLMakie
using Makie.GeometryBasics

@views function PlotMakie( mesh, v )
    ``` Create patch plots for either Quadrangles or Triangles
    ```
    f = Figure()

    Axis(f[1, 1])

    p = [Polygon( Point2f0[ (mesh.xv[mesh.e2v[i,j]], mesh.yv[mesh.e2v[i,j]]) for j=1:mesh.nf_el] ) for i in 1:mesh.nel]

    poly!(p, color = v, colormap = :viridis, strokewidth = 0, strokecolor = :black, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect_ratio = :equal, clims=[0.7 1.4])

    Colorbar(f[1, 2], colormap = :viridis, limits=[0.7 1.4], flipaxis = true, size = 25, height = Relative(2/3) )

    xlims!(0, 1)
    ylims!(0, 1)

    display(f)
    return 
end
