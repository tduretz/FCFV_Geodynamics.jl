include("CreateMesh.jl")
using LoopVectorization
using CairoMakie
import AbstractPlotting.GeometryBasics
using SparseArrays, LinearAlgebra
import UnicodePlots 
import Plots


# # ENV["MPLBACKEND"]="Qt5Agg"
# # ENV["MPLBACKEND"] = "module://gr.matplotlib.backend_gr"
# # Problem #1 was to find a simple way to plot piecewise constant fields on triangles:
# # solution from: https://robertsweeneyblanco.github.io/Programming_for_Mathematical_Applications/Computational_Geometry/Triangulations.html
# # Problem #2 was to get any figures popping out in VScode 
# # and: https://github.com/JuliaPy/PyPlot.jl/issues/418
# # import PyCall, PyPlot
# using Base.Threads
# using LoopVectorization
# import TriangleMesh
# using Printf
# # PyCall.pygui(:qt5)
# # PyPlot.pygui(true)

#--------------------------------------------------------------------#

# function tplot(mesh, v)

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
#     PyPlot.clf()
#     tris =  PyPlot.convert(Array{Int64}, hcat(t...)')
#     display(PyPlot.tripcolor(first.(p), last.(p), tris .- 1, v,
#               cmap="viridis", edgecolors="none", linewidth=0) )
#     PyPlot.axis("equal")
#     PyPlot.ylim([0, 1])
#     PyPlot.xlim([0, 1])
#     PyPlot.title("Low res.")
#     PyPlot.xlabel("x")
#     PyPlot.ylabel("y")
#     PyPlot.colorbar()
#     PyPlot.show()
#     return 
# end

@views function PlotMakie( mesh, v )
    p   = [AbstractPlotting.GeometryBasics.Polygon( Point2f0[ (mesh.xv[mesh.e2v[i,j]], mesh.yv[mesh.e2v[i,j]]) for j=1:mesh.nf_el] ) for i in 1:mesh.nel]
    scene = poly(p, color = v, colormap = :jet1, strokewidth = 0, strokecolor = :black, markerstrokewidth=0, markerstrokecolor = (0, 0, 0, 0), aspect_ratio=:equal, clims=[0.7 1.4])
    display(scene)

    # red = 20
    # iel = 100
    # p = Plots.plot()
    # for iel=1:10#mesh.nel
    #         Plots.plot!(  [mesh.xv[mesh.e2v[iel,2]], mesh.xv[mesh.e2v[iel,3]]], [mesh.yv[mesh.e2v[iel,2]], mesh.yv[mesh.e2v[iel,3]]], leg = false, aspect_ratio=:equal   )
    #         Plots.plot!( [mesh.xv[mesh.e2v[iel,3]], mesh.xv[mesh.e2v[iel,1]]], [mesh.yv[mesh.e2v[iel,3]], mesh.yv[mesh.e2v[iel,1]]]  )
    #         Plots.plot!( [mesh.xv[mesh.e2v[iel,1]], mesh.xv[mesh.e2v[iel,2]]], [mesh.yv[mesh.e2v[iel,1]], mesh.yv[mesh.e2v[iel,2]]]  )

    
    #     Plots.scatter!( [mesh.xf[mesh.e2f[iel,1]]], [mesh.yf[mesh.e2f[iel,1]]]  )    
    #     Plots.plot!(    [mesh.xf[mesh.e2f[iel,1]], mesh.xf[mesh.e2f[iel,1]]+mesh.n_x[iel,1]/red], [mesh.yf[mesh.e2f[iel,1]], mesh.yf[mesh.e2f[iel,1]]+mesh.n_y[iel,1]/red ] )
    #     Plots.scatter!( [mesh.xf[mesh.e2f[iel,2]]], [mesh.yf[mesh.e2f[iel,2]]]  )    
    #     Plots.plot!(    [mesh.xf[mesh.e2f[iel,2]], mesh.xf[mesh.e2f[1iel,2]]+mesh.n_x[iel,2]/red], [mesh.yf[mesh.e2f[iel,2]], mesh.yf[mesh.e2f[iel,2]]+mesh.n_y[iel,2]/red ] )
    #     Plots.scatter!( [mesh.xf[mesh.e2f[iel,3]]], [mesh.yf[mesh.e2f[iel,3]]]  )    
    #     Plots.plot!(    [mesh.xf[mesh.e2f[iel,3]], mesh.xf[mesh.e2f[iel,3]]+mesh.n_x[iel,3]/red], [mesh.yf[mesh.e2f[iel,3]], mesh.yf[mesh.e2f[iel,3]]+mesh.n_y[iel,3]/red ] )
    # end
    # display(p)
    return 
end

function Tanalytic2!( mesh, T, Tdir, Tneu, se, a, b, c, d, alp, bet )
    # Evaluate T analytic on cell faces
    @avx for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        Tdir[in] = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
    end
    # Evaluate T analytic on barycentres
    @avx for iel=1:mesh.nel
        x       = mesh.xc[iel]
        y       = mesh.yc[iel]
        T       = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        T[iel]  = T
        se[iel] = T*(-a*alp*cos(a*x + c*y) + b*bet*sin(b*x + d*y))*(a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y)) + T*(a^2*alp*sin(a*x + c*y) + b^2*bet*cos(b*x + d*y)) + T*(-alp*c*cos(a*x + c*y) + bet*d*sin(b*x + d*y))*(alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y)) + T*(alp*c^2*sin(a*x + c*y) + bet*d^2*cos(b*x + d*y))
    end
    return
end

function StabParam(tau, dA, Vol)
    taui = tau*dA
    return taui
end

function ComputeFCFV(mesh, se, Tdir, tau)
    ae = zeros(mesh.nel)
    be = zeros(mesh.nel)
    ze = zeros(mesh.nel,2)

    # Assemble FCFV elements
    @avx for iel=1:mesh.nel  

        be[iel] = be[iel]   + mesh.vole[iel]*se[iel]
        
        for ifac=1:mesh.nf_el
            
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel])                              # Stabilisation parameter for the face

            # Assemble
            ze[iel,1] += (bc==1) * dAi*ni_x*Tdir[nodei]  # Dirichlet
            ze[iel,2] += (bc==1) * dAi*ni_y*Tdir[nodei]  # Dirichlet
            be[iel]   += (bc==1) * dAi*taui*Tdir[nodei]  # Dirichlet
            ae[iel]   +=           dAi*taui
            
        end
    end
    return ae, be, ze
end

function ComputeElementValues(mesh, uh, ae, be, ze, Tdir, tau)
    ue          = zeros(mesh.nel);
    qx          = zeros(mesh.nel);
    qy          = zeros(mesh.nel);

    @avx for iel=1:mesh.nel
    
        ue[iel]  =  be[iel]/ae[iel]
        qx[iel]  = -1.0/mesh.vole[iel]*ze[iel,1]
        qy[iel]  = -1.0/mesh.vole[iel]*ze[iel,2]
        
        for ifac=1:mesh.nf_el
            
            # Face
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel])      # Stabilisation parameter for the face

            # Assemble
            ue[iel] += (bc!=1) *  dAi*taui*uh[mesh.e2f[iel, ifac]]/ae[iel]
            qx[iel] -= (bc!=1) *  1.0/mesh.vole[iel]*dAi*ni_x*uh[mesh.e2f[iel, ifac]]
            qe[iel] -= (bc!=1) *  1.0/mesh.vole[iel]*dAi*ni_y*uh[mesh.e2f[iel, ifac]]
         end
    end
    return ue, qx, qy
end

@views function main()

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = 80, 80
    mesh_type  = "Quadrangles"
    # mesh_type  = "UnstructTriangles"
  
    if mesh_type=="Quadrangles" 
        tau = 100
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax )
    elseif mesh_type=="UnstructTriangles"  
        tau = 100
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax ) 
    end

    println("Number of elements: ", mesh.nel)

    # Source term and BCs etc...
    Tanal  = zeros(mesh.nel)
    se     = zeros(mesh.nel)
    Tdir   = zeros(mesh.nf)
    Tneu   = zeros(mesh.nf)
    alp = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;
    @time Tanalytic2!(mesh , Tanal, Tdir, Tneu, se, a, b, c, d, alp, bet)

    # Compute some mesh vectors 
    ae, be, ze = ComputeFCFV(mesh, se, Tdir, tau)

    # Assemble stiffness matrix 
    rows = Int64[]
    cols = Int64[]
    vals = Float64[]
    f    = zeros(mesh.nf);

    for iel=1:mesh.nel 

        Ke = zeros(mesh.nf_el,mesh.nf_el);
        fe = zeros(mesh.nf_el,1);
        
        for ifac=1:mesh.nf_el 

            nodei  = mesh.e2f[iel,ifac]
            bci    = mesh.bc[nodei]
            
            if bci != 1

                dAi  = mesh.dA[iel,ifac]
                ni_x = mesh.n_x[iel,ifac]
                ni_y = mesh.n_y[iel,ifac]
                taui = StabParam(tau,dAi,mesh.vole[iel])  
                
                for jfac=1:mesh.nf_el

                    nodej  = mesh.e2f[iel,jfac]
                    bcj    = mesh.bc[nodej]
                    
                    if bcj!= 1
                        
                        dAj  = mesh.dA[iel,jfac]
                        nj_x = mesh.n_x[iel,jfac]
                        nj_y = mesh.n_y[iel,jfac]
                        tauj = StabParam(tau,dAj,mesh.vole[iel])  
                        
                        # Delta
                        del = 0.0
                        if ifac==jfac; del = 1.0; end
                        
                        # Element matrix
                        nitnj         = ni_x*nj_x + ni_y*nj_y;
                        Ke[ifac,jfac] = dAi * (1.0/ae[iel] * dAj * taui*tauj - 1.0/mesh.vole[iel]*dAj*nitnj - taui*del);
                    end
                    
                end

                # RHS vector
                Xi = 0.0;
                if bci == 2; Xi = 1.0; end # indicates Neumann dof
                ti = Tneu[nodei]
                nitze     = ni_x*ze[iel,1] + ni_y*ze[iel,2]
                fe[ifac]  = dAi * (1.0/mesh.vole[iel]*nitze - ti*Xi - 1.0/ae[iel]*be[iel]*taui)
                
            end
        end
        # println(Ke)

        for ifac=1:mesh.nf_el

            nodei  = mesh.e2f[iel,ifac]
            bci    = mesh.bc[nodei]
            
            if bci != 1

                for jfac=1:mesh.nf_el

                    nodej  = mesh.e2f[iel,jfac]
                    bcj    = mesh.bc[nodej]

                    if bcj != 1
                        push!(rows, mesh.e2f[ iel,ifac])  
                        push!(cols, mesh.e2f[ iel,jfac]) 
                        push!(vals,      -Ke[ifac,jfac])
                    end
                end
                f[mesh.e2f[ iel,ifac]] -= fe[ifac]
            else
                push!(rows, mesh.e2f[ iel,ifac])  
                push!(cols, mesh.e2f[ iel,ifac]) 
                push!(vals,                 1.0)
                f[mesh.e2f[ iel,ifac]] = Tdir[nodei]
            end
        end

    end
    K = sparse(rows, cols, vals, mesh.nf, mesh.nf)
    droptol!(K, 1e-6)
    uh   = K\f

    # Reconstruct element values
    Te, qx, qy = ComputeElementValues(mesh, uh, ae, be, ze, Tdir, tau)
    
    # display(UnicodePlots.spy(K))
    # println(uh[:])

    # for iel=1:mesh.nel
    #     println(ue[iel])
    # end

    # for i=1:length(uh)
    #     println(uh[i])
    # end

    # println(mesh.e2f)


    # import MAT
    # file   = MAT.matopen("/Users/imac/ownCloud/FCFV/Mat100_ue.mat")
    # ue_mat = MAT.read(file, "ue")

    # Visualise
    @time PlotMakie( mesh, Te )
    println(maximum(Te))
    println(maximum(Tanal))
    # tplot(mesh, ue)
    # function qplot(x, y, v)
    # xc=LinRange(xmin,xmax,nx)
    # yc=LinRange(ymin,ymax,ny)
    # display( heatmap(xc,yc,reshape(ue,nx,ny), clim=[0.7 1.5]) )
        # clf()
        # display( pcolor(xc, yc, reshape(ue,nx,ny)) )
        # colorbar()
        # xlabel("x")
        # ylabel("y")
        # show()
    # end

end

main()