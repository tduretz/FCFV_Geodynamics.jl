using Printf, Plots, LinearAlgebra

include("FunctionsFEM.jl")
include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../SolversFCFV_Stokes.jl")
include("IntegrationPoints.jl")
include("Helpers.jl")

#-----------------------------------------------------------------#

@views function main( n, nnel, npel, nip, θ, ΔτV, ΔτP )

    println("\n******** FEM STOKES ********")
    g      = [0.0 -1.0]
    ρ      = 1
    η      = 1.0
    solver =-1

    xmin, xmax = -3.0, 3.0
    ymin, ymax = -5.0, 0.0
    R          = 1.0
    model      = -2
    nx, ny     = Int16(n*30), Int16(n*30)
    #-----------------------------------------------------------------#
    
    # Generate
    BC   = [1; 1; 2; 1;] # S E N W --- 1: Dirichlet / 2: Neumann
    mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, model, R, BC; nnel, npel )     
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    println("Number of p nodes:  ", mesh.np)
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    println("Number of p nodes:  ", mesh.np)
    println("Min. bcn:  ", minimum(mesh.bcn))
    println("Max. bcn:  ", maximum(mesh.bcn))

    mesh.ke .= η
  
    #-----------------------------------------------------------------#
    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)

    #-----------------------------------------------------------------#
    Vx  = zeros(mesh.nn)       # Solution on nodes 
    Vy  = zeros(mesh.nn)       # Solution on nodes 
    P   = zeros(mesh.np)       # Solution on either elements or nodes 
    se  = zeros(mesh.nel,2)    # Source on elements

    se[:,1]    .= ρ*g[1]
    se[:,2]    .= ρ*g[2]

    
    #-----------------------------------------------------------------#
    @time Kuu, Kup, bu, bp = ElementAssemblyLoopFEM_v2( se, mesh, ipx, ipw, N, dNdX, Vx, Vy, P )
    
    #-----------------------------------------------------------------#
    coef  = zeros(mesh.nel*mesh.npel)
    Kpp   = spdiagm(coef)
    @time StokesSolvers!(Vx, Vy, P, mesh, Kuu, Kup, -Kup', Kpp, bu, bp, Kuu, solver; penalty=1e2)
    
    #-----------------------------------------------------------------#
    Vxe = zeros(mesh.nel)
    Vye = zeros(mesh.nel)
    Ve  = zeros(mesh.nel)
    Pe  = zeros(mesh.nel)
    for e=1:mesh.nel
        for i=1:mesh.nnel
            Vxe[e] += 1.0/mesh.nnel * Vx[mesh.e2n[e,i]]
            Vye[e] += 1.0/mesh.nnel * Vy[mesh.e2n[e,i]]
            Ve[e]  += 1.0/mesh.nnel * sqrt(Vx[mesh.e2n[e,i]]^2 + Vy[mesh.e2n[e,i]]^2)
        end
        for i=1:mesh.npel
            Pe[e] += 1.0/mesh.npel * P[mesh.e2p[e,i]]
        end
    end
    pminmax(Vx)
    pminmax(Vy)
    pminmax(P)

    #-----------------------------------------------------------------#
    @time PlotMakie(mesh, Vxe, minimum(mesh.xn), maximum(mesh.xn), minimum(mesh.yn), maximum(mesh.yn); min_v=-0.055, max_v=0.055, cmap=:turbo, write_fig=false)
    # @time PlotMakie(mesh, Vxe, minimum(mesh.xn), maximum(mesh.xn), minimum(mesh.yn), maximum(mesh.yn); min_v=-0.055, max_v=0.055, cmap=:turbo, write_fig=true)
    # @time PlotMakie(mesh, Vye, minimum(mesh.xn), maximum(mesh.xn), minimum(mesh.yn), maximum(mesh.yn); min_v=-0.2, max_v=0.1, cmap=:turbo, write_fig=true)
    # @time PlotMakie(mesh, Pe, minimum(mesh.xn), maximum(mesh.xn), minimum(mesh.yn), maximum(mesh.yn);  min_v=-0.0, max_v=5., cmap=:turbo, write_fig=true)
    #-----------------------------------------------------------------#
end

for i=1:1
   main( 2, 7, 1, 6, 0., 0., 0. )
end