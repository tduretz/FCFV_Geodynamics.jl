using CairoMakie
import AbstractPlotting.GeometryBasics
import Plots

@views function PlotMakie( mesh, v )
    ``` Create patch plots for either Quadrangles or Triangles
    ```
    p   = [AbstractPlotting.GeometryBasics.Polygon( Point2f0[ (mesh.xv[mesh.e2v[i,j]], mesh.yv[mesh.e2v[i,j]]) for j=1:mesh.nf_el] ) for i in 1:mesh.nel]
    scene = poly(p, color = v, colormap = :jet1, strokewidth = 0, strokecolor = :black, markerstrokewidth=0, markerstrokecolor = (0, 0, 0, 0), aspect_ratio=:equal, clims=[0.7 1.4])
    display(scene)
    return 
end

@views function PlotElements( mesh )
    ``` Only works for triangles so far
    ```
    red = 20
    iel = 100
    p = Plots.plot()
    for iel=1:10#mesh.nel
        # Element edges
        Plots.plot!( [mesh.xv[mesh.e2v[iel,2]], mesh.xv[mesh.e2v[iel,3]]], [mesh.yv[mesh.e2v[iel,2]], mesh.yv[mesh.e2v[iel,3]]], leg = false, aspect_ratio=:equal   )
        Plots.plot!( [mesh.xv[mesh.e2v[iel,3]], mesh.xv[mesh.e2v[iel,1]]], [mesh.yv[mesh.e2v[iel,3]], mesh.yv[mesh.e2v[iel,1]]]  )
        Plots.plot!( [mesh.xv[mesh.e2v[iel,1]], mesh.xv[mesh.e2v[iel,2]]], [mesh.yv[mesh.e2v[iel,1]], mesh.yv[mesh.e2v[iel,2]]]  )
        # Element normals
        Plots.scatter!( [mesh.xf[mesh.e2f[iel,1]]], [mesh.yf[mesh.e2f[iel,1]]]  )    
        Plots.plot!(    [mesh.xf[mesh.e2f[iel,1]], mesh.xf[mesh.e2f[iel,1]]+mesh.n_x[iel,1]/red], [mesh.yf[mesh.e2f[iel,1]], mesh.yf[mesh.e2f[iel,1]]+mesh.n_y[iel,1]/red ] )
        Plots.scatter!( [mesh.xf[mesh.e2f[iel,2]]], [mesh.yf[mesh.e2f[iel,2]]]  )    
        Plots.plot!(    [mesh.xf[mesh.e2f[iel,2]], mesh.xf[mesh.e2f[1iel,2]]+mesh.n_x[iel,2]/red], [mesh.yf[mesh.e2f[iel,2]], mesh.yf[mesh.e2f[iel,2]]+mesh.n_y[iel,2]/red ] )
        Plots.scatter!( [mesh.xf[mesh.e2f[iel,3]]], [mesh.yf[mesh.e2f[iel,3]]]  )    
        Plots.plot!(    [mesh.xf[mesh.e2f[iel,3]], mesh.xf[mesh.e2f[iel,3]]+mesh.n_x[iel,3]/red], [mesh.yf[mesh.e2f[iel,3]], mesh.yf[mesh.e2f[iel,3]]+mesh.n_y[iel,3]/red ] )
    end
    display(p)
end