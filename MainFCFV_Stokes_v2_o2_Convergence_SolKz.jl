import TriangleMesh, UnicodePlots, Plots
using Revise, Printf, LoopVectorization, LinearAlgebra, SparseArrays

include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV_Stokes.jl")
include("DiscretisationFCFV_Stokes_o2.jl")
include("SolversFCFV_Stokes.jl")
include("SolKz_JustRelax.jl")

#--------------------------------------------------------------------#
    
function StabParam(tau, dA, Vol, mesh_type, coeff)
    if mesh_type=="Quadrangles";        taui = tau;    end
    # if mesh_type=="Quadrangles";        taui = 500*tau/Vol;    end
    # if mesh_type=="UnstructTriangles";  taui = tau*dA; end
    if mesh_type=="UnstructTriangles";  taui = tau*1.0 end
    return taui
end

#--------------------------------------------------------------------#

function SetUpProblem!(mesh, P, Vx, Vy, Sxx, Syy, Sxy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, sx, sy)
    # Evaluate T analytic on cell faces
    for in=1:mesh.nf
        x         = mesh.xf[in]
        y         = mesh.yf[in]
        vx, vy, p, ρ, η = _solkz_solution(x, y)
        eta       = η
        VxDir[in] = vx
        VyDir[in] = vy
        # Stress at faces using pseudo-tractions
        dVxdx = 0.0
        dVxdy = 0.0
        dVydx = 0.0
        dVydy = 0.0
        SxxNeu[in] = - p + eta*dVxdx 
        SyyNeu[in] = - p + eta*dVydy 
        SxyNeu[in] =       eta*dVxdy
        SyxNeu[in] =       eta*dVydx
    end
    # Evaluate T analytic on barycentres
    for iel=1:mesh.nel
        x        = mesh.xc[iel]
        y        = mesh.yc[iel]
        vx, vy, p, ρ, η = _solkz_solution(x, y)
        eta      = η
        mesh.ke[iel] = eta/2.0
        P[iel]   =  p
        Vx[iel]  =  vx
        Vy[iel]  =  vy
        Sxx[iel] = 0.0
        Syy[iel] = 0.0
        Sxy[iel] = 0.0
        sx[iel]  = 0.0
        sy[iel]  = -ρ
    end
    return
end

#--------------------------------------------------------------------#

function ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe )
    eVx  = zeros(mesh.nel)
    eVy  = zeros(mesh.nel)
    eTxx = zeros(mesh.nel)
    eTyy = zeros(mesh.nel)
    eTxy = zeros(mesh.nel)
    eP   = zeros(mesh.nel)
    eV   = zeros(mesh.nel)
    eTii = zeros(mesh.nel)
    Vxa  = zeros(mesh.nel)
    Vya  = zeros(mesh.nel)
    Txxa = zeros(mesh.nel)
    Tyya = zeros(mesh.nel)
    Txya = zeros(mesh.nel)
    Pa   = zeros(mesh.nel)
    Tiia = zeros(mesh.nel)
    Va   = zeros(mesh.nel)
    @avx for iel=1:mesh.nel
        x         = mesh.xc[iel]
        y         = mesh.yc[iel]
        vx, vy, p, ρ, η = _solkz_solution(x, y)
        Pa[iel]   =  p
        Vxa[iel]  =  vx
        Vya[iel]  =  vy
        Txxa[iel] =  0.0
        Tyya[iel] =  0.0
        Txya[iel] =  0.0
        eVx[iel]  = Vxe[iel]  - Vxa[iel]
        eVy[iel]  = Vye[iel]  - Vya[iel]
        eTxx[iel] = Txxe[iel] - Txxa[iel]
        eTyy[iel] = Tyye[iel] - Tyya[iel]
        eTxy[iel] = Txye[iel] - Txya[iel]
        eP[iel]   = Pe[iel]   - Pa[iel]
        Va[iel]   = sqrt(Vxa[iel]^2 + Vya[iel]^2) 
        Ve        = sqrt(Vxe[iel]^2 + Vye[iel]^2)
        Tiia[iel] = sqrt(0.5*(Txxa[iel]^2 + Tyya[iel]^2) + Txya[iel]^2)
        Tiie      = sqrt(0.5*(Txxe[iel]^2 + Tyye[iel]^2) + Txye[iel]^2)
        eV[iel]   = Ve   - Va[iel]
        eTii[iel] = Tiie - Tiia[iel] 
    end
    errVx  = norm(eVx) /norm(Vxa)
    errVy  = norm(eVy) /norm(Vya)
    errTxx = norm(eTxx)/norm(Txxa)
    errTyy = norm(eTyy)/norm(Tyya)
    errTxy = norm(eTxy)/norm(Txya)
    errP   = norm(eP)  /norm(Pa)
    errV   = norm(eV)  /norm(Va)
    errTii = norm(eTii)/norm(Tiia)
    return errVx, errVy, errTxx, errTyy, errTxy, errP, errV, errTii
end

#--------------------------------------------------------------------#

@views function main( N, mesh_type, order, new )

    println("\n******** FCFV STOKES ********")

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    n          = 4
    nx, ny     = N, N
    solver     = 0
    R          = 0.5
    inclusion  = 0
    BC         = [1; 1; 1; 1] # S E N W --- 1: Dirichlet / 2: Neumann
    o2         = order-1
    # mesh_type  = "Quadrangles"
    # mesh_type  = "UnstructTriangles"
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        if o2==0 tau  = 2e1 end
        if o2==1 tau  = 1e6 end
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, tau, inclusion, R, BC )
    elseif mesh_type=="UnstructTriangles"  
        if o2==0 tau  = 2e1 end
        if o2==1 tau  = 20e4 end
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, tau, inclusion, R, BC ) 
    end
    println("Number of elements: ", mesh.nel)

    # Source term and BCs etc...
    Pa     = zeros(mesh.nel)
    Vxa    = zeros(mesh.nel)
    Vya    = zeros(mesh.nel)
    Sxxa   = zeros(mesh.nel)
    Syya   = zeros(mesh.nel)
    Sxya   = zeros(mesh.nel)
    sex    = zeros(mesh.nel)
    sey    = zeros(mesh.nel)
    VxDir  = zeros(mesh.nf)
    VyDir  = zeros(mesh.nf)
    SxxNeu = zeros(mesh.nf)
    SyyNeu = zeros(mesh.nf)
    SxyNeu = zeros(mesh.nf)
    SyxNeu = zeros(mesh.nf)
    gbar   = zeros(mesh.nel,mesh.nf_el,2)
    println("Model configuration :")
    @time SetUpProblem!(mesh, Pa, Vxa, Vya, Sxxa, Syya, Sxya, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, sex, sey)

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    @time ae, be, be_o2, ze, pe, mei, pe, rjx, rjy = ComputeFCFV_o2(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, o2)

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    @time Kuu, Muu, Kup, fu, fp, tsparse = ElementAssemblyLoop_o2(mesh, ae, be, be_o2, ze, mei, pe, rjx, rjy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, o2, new)

    # Solve for hybrid variable
    println("Linear solve:")
    @time Vxh, Vyh, Pe = StokesSolvers(mesh, Kuu, Kup, fu, fp, Muu, solver)

    # Reconstruct element values
    println("Compute element values:")
    @time Vxe, Vye, Txxe, Tyye, Txye = ComputeElementValues_o2(mesh, Vxh, Vyh, Pe, ae, be, be_o2, ze, rjx, rjy, mei, VxDir, VyDir, o2)

    # # Compute discretisation errors
    err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii = ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe )
    @printf("Error in Vx : %2.2e\n", err_Vx )
    @printf("Error in Vy : %2.2e\n", err_Vy )
    @printf("Error in Txx: %2.2e\n", err_Txx)
    @printf("Error in Tyy: %2.2e\n", err_Tyy)
    @printf("Error in Txy: %2.2e\n", err_Txy)
    @printf("Error in P  : %2.2e\n", err_P  )

    # Visualise
    # println("Visualisation:")
    @printf(" %2.2e %2.2e\n", minimum(log10.(mesh.ke)), maximum(log10.(mesh.ke)))
    @printf(" %2.2e %2.2e\n", minimum(sey), maximum(sey))
    # @time PlotMakie( mesh, log10.(mesh.ke), xmin, xmax, ymin, ymax, cgrad(:lajolla, rev=true) )
    # @time PlotMakie( mesh, Vye, xmin, xmax, ymin, ymax, cgrad(:lajolla, rev=true) )
    # @time PlotMakie( mesh, sey, xmin, xmax, ymin, ymax, cgrad(:lajolla, rev=true) )
    @time PlotMakie( mesh, Pe, xmin, xmax, ymin, ymax, cgrad(:lajolla, rev=true) )

    # PlotElements( mesh )
    ndof = 2*mesh.nf+mesh.nel
    return ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii
end

function Run()
    order      = 1 
    new        = 0
    N          = 64
    mesh_type  = "UnstructTriangles"
    # mesh_type  = "Quadrangles"
    @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii  = main( N, mesh_type, order, new )
end



function RunConvergence()

    #################### ORDER 1
    order = 1 

    N             = [8, 16, 32, 64, 128]#, 256], 512, 1024]
    mesh_type  = "Quadrangles"
    eV_quad    = zeros(size(N))
    eP_quad    = zeros(size(N))
    eTau_quad  = zeros(size(N))
    t_quad     = zeros(size(N))
    ndof_quad  = zeros(size(N))
    for k=1:length(N)
        t_quad[k]    = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii = main( N[k], mesh_type, order )
        eV_quad[k]   = err_V
        eP_quad[k]   = err_P
        eTau_quad[k] = err_Tii
        ndof_quad[k] = ndof
    end

    mesh_type  = "UnstructTriangles"
    eV_tri     = zeros(size(N))
    eP_tri     = zeros(size(N))
    eTau_tri   = zeros(size(N))
    t_tri      = zeros(size(N))
    ndof_tri   = zeros(size(N))
    for k=1:length(N)
        t_tri[k]     = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii  = main( N[k], mesh_type, order )
        eV_tri[k]    = err_V
        eP_tri[k]    = err_P
        eTau_tri[k]  = err_Tii
        ndof_tri[k]  = ndof
    end

    #################### ORDER 2
    order = 2 

    mesh_type     = "Quadrangles"
    eV_quad_o2    = zeros(size(N))
    eP_quad_o2    = zeros(size(N))
    eTau_quad_o2  = zeros(size(N))
    t_quad_o2     = zeros(size(N))
    ndof_quad_o2  = zeros(size(N))
    for k=1:length(N)
        t_quad_o2[k]    = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii = main( N[k], mesh_type, order )
        eV_quad_o2[k]   = err_V
        eP_quad_o2[k]   = err_P
        eTau_quad_o2[k] = err_Tii
        ndof_quad_o2[k] = ndof
    end

    mesh_type     = "UnstructTriangles"
    eV_tri_o2     = zeros(size(N))
    eP_tri_o2     = zeros(size(N))
    eTau_tri_o2   = zeros(size(N))
    t_tri_o2      = zeros(size(N))
    ndof_tri_o2   = zeros(size(N))
    for k=1:length(N)
        t_tri_o2[k]     = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii  = main( N[k], mesh_type, order )
        eV_tri_o2[k]    = err_V
        eP_tri_o2[k]    = err_P
        eTau_tri_o2[k]  = err_Tii
        ndof_tri_o2[k]  = ndof
    end

    # #######################################

    p = Plots.plot(  log10.(1.0 ./ N) , log10.(eV_quad),   markershape=:rect,      color=:blue,                         label="Quads V O1"                          )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_quad),   markershape=:rect,      color=:blue,      linestyle = :dot,  label="Quads P O1"                          )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_quad), markershape=:rect,      color=:blue,      linestyle = :dash, label="Quads Tau O1"                        )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eV_tri),    markershape=:dtriangle, color=:blue,                         label="Triangles V O1"                     )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_tri),    markershape=:dtriangle, color=:blue,      linestyle = :dot,  label="Triangles P O1"                     )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_tri),  markershape=:dtriangle, color=:blue,      linestyle = :dash, label="Triangles Tau O1")#, legend=:bottomright, xlabel = "log_10(h_x)", ylabel = "log_10(err_T)" )

    p = Plots.plot!(  log10.(1.0 ./ N) , log10.(eV_quad_o2),   markershape=:rect,      color=:red,                         label="Quads V O2"                          )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_quad_o2),   markershape=:rect,      color=:red,      linestyle = :dot,  label="Quads P O2"                          )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_quad_o2), markershape=:rect,      color=:red,      linestyle = :dash, label="Quads Tau O2"                        )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eV_tri_o2),    markershape=:dtriangle, color=:red,                         label="Triangles V O2"                     )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_tri_o2),    markershape=:dtriangle, color=:red,      linestyle = :dot,  label="Triangles P O2"                     )
    p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_tri_o2),  markershape=:dtriangle, color=:red,      linestyle = :dash, label="Triangles Tau O2", legend=:outertopright, xlabel = "log_10(h_x)", ylabel = "log_10(err_T)" )

    order1 = [2e-4, 1e-4]
    order2 = [4e-4, 1e-4]
    n      = [10, 20]
    p = Plots.plot!( log10.(1.0 ./ n) , log10.(order1), color=:black, label="Order 1")
    p = Plots.plot!( log10.(1.0 ./ n) , log10.(order2), color=:black, label="Order 2", linestyle = :dash)
    p = Plots.annotate!(log10.(1.0 ./ N[1]), log10(order1[1]), "O1", :black)
    p = Plots.annotate!(log10.(1.0 ./ N[1]), log10(order2[1]), "O2", :black)
    # p = Plots.plot(  ndof_quad[2:end], t_quad[2:end], markershape=:rect,      label="Quads"                          )
    # p = Plots.plot!( ndof_tri[2:end],  t_tri[2:end],  markershape=:dtriangle, label="Triangles", legend=:bottomright, xlabel = "ndof", ylabel = "time" )
    display(p)

end 

Run()
# RunConvergence()