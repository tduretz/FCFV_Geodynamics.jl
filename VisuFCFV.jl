using GLMakie
using Makie.GeometryBasics

@views function PlotMakie( mesh, v, xmin, xmax, ymin, ymax, cmap, cblims )
    ``` Create patch plots for either Quadrangles or Triangles
    ```

    min_v, max_v = minimum(v), maximum(v)

    f = Figure()

    Axis(f[1, 1])

    p = [Polygon( Point2f0[ (mesh.xv[mesh.e2v[i,j]], mesh.yv[mesh.e2v[i,j]]) for j=1:mesh.nf_el] ) for i in 1:mesh.nel]

    poly!(p, color = v, colormap = cmap, strokewidth = 0, strokecolor = :black, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect_ratio = :equal, clims=cblims)

    Colorbar(f[1, 2], colormap = cmap, limits=cblims, flipaxis = true, size = 25, height = Relative(2/3) )
    
    # poly!(p, color = v, colormap = cmap, strokewidth = 0, strokecolor = :black, markerstrokewidth = 0, markerstrokecolor = (0, 0, 0, 0), aspect_ratio = :equal, clims=cblims)

    # Colorbar(f[1, 2], colormap = cmap, limits=cblims, flipaxis = true, size = 25, height = Relative(2/3) )


    xlims!(xmin, xmax)
    ylims!(ymin, ymax)

    display(f)
    return 
end
