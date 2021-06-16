include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV_Stokes.jl")
include("SolversFCFV_Stokes.jl")
using LoopVectorization, Printf
using SparseArrays, LinearAlgebra
import UnicodePlots 

#--------------------------------------------------------------------#

function SetUpProblem!(mesh, P, Vx, Vy, Sxx, Syy, Sxy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, sx, sy)
    # Evaluate T analytic on cell faces
    @avx for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        VxDir[in] = x^2*(1 - x)^2*(4*y^3 - 6*y^2 + 2*y)
        VyDir[in] =-y^2*(1 - y)^2*(4*x^3 - 6*x^2 + 2*x)
        # Stress at faces
        p          =  x*(1-x)
        SxxNeu[in] = -8*p*y*(x - 1)*(2*y^2 - 3*y + 1) - p + 8*x^2*y*(x - 1)*(2*y^2 - 3*y + 1)
        SyyNeu[in] = -p - 8*x*y^2*(y - 1)*(2*x^2 - 3*x + 1) - 8*x*y*(y - 1)^2*(2*x^2 - 3*x + 1)
        SxyNeu[in] = p^2*(12*y^2 - 12*y + 2) + y^2*(y - 1)^2*(-12.0*x^2 + 12.0*x - 2.0)
    end
    # Evaluate T analytic on barycentres
    @avx for iel=1:mesh.nel
        x        = mesh.xc[iel]
        y        = mesh.yc[iel]
        p        =  x*(1-x)
        P[iel]   =  p
        Vx[iel]  =  x^2*(1 - x)^2*(4*y^3 - 6*y^2 + 2*y)
        Vy[iel]  = -y^2*(1 - y)^2*(4*x^3 - 6*x^2 + 2*x)
        Sxx[iel] = -8*p*y*(x - 1)*(2*y^2 - 3*y + 1) - p + 8*x^2*y*(x - 1)*(2*y^2 - 3*y + 1)
        Syy[iel] = -p - 8*x*y^2*(y - 1)*(2*x^2 - 3*x + 1) - 8*x*y*(y - 1)^2*(2*x^2 - 3*x + 1)
        Sxy[iel] = p^2*(12*y^2 - 12*y + 2) + y^2*(y - 1)^2*(-12.0*x^2 + 12.0*x - 2.0)
        sx[iel]  = -p^2*(24*y - 12) - 4*x^2*(4*y^3 - 6*y^2 + 2*y) - 8*x*(2*x - 2)*(4*y^3 - 6*y^2 + 2*y) - 2*x + 1.0*y^2*(2*y - 2)*(12*x^2 - 12*x + 2) + 2.0*y*(1 - y)^2*(12*x^2 - 12*x + 2) - 4*(1 - x)^2*(4*y^3 - 6*y^2 + 2*y) + 1
        sy[iel]  = -2*p*(1 - x)*(12*y^2 - 12*y + 2) - x^2*(2*x - 2)*(12*y^2 - 12*y + 2) + 1.0*y^2*(1 - y)^2*(24*x - 12) + 4*y^2*(4*x^3 - 6*x^2 + 2*x) + 8*y*(2*y - 2)*(4*x^3 - 6*x^2 + 2*x) + 4*(1 - y)^2*(4*x^3 - 6*x^2 + 2*x)
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
        Pa[iel]   =  x*(1-x)
        Vxa[iel]  =  x^2*(1 - x)^2*(4*y^3 - 6*y^2 + 2*y)
        Vya[iel]  = -y^2*(1 - y)^2*(4*x^3 - 6*x^2 + 2*x)
        Txxa[iel] = -8*Pa[iel]*y*(x - 1)*(2*y^2 - 3*y + 1) + 8*x^2*y*(x - 1)*(2*y^2 - 3*y + 1)
        Tyya[iel] = -8*x*y^2*(y - 1)*(2*x^2 - 3*x + 1) - 8*x*y*(y - 1)^2*(2*x^2 - 3*x + 1)
        Txya[iel] = Pa[iel]^2*(12*y^2 - 12*y + 2) + y^2*(y - 1)^2*(-12.0*x^2 + 12.0*x - 2.0)
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
    
function StabParam(tau, dA, Vol, mesh_type)
    if mesh_type=="Quadrangles";        taui = tau;    end
    # if mesh_type=="UnstructTriangles";  taui = tau*dA; end
    if mesh_type=="UnstructTriangles";  taui = tau end
    return taui
end

#--------------------------------------------------------------------#
    
@views function main( N, mesh_type )

    println("\n******** FCFV STOKES ********")

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = N, N
    solver     = 1
    R          = 0.5
    inclusion  = 0
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        tau  = 1
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R )
    elseif mesh_type=="UnstructTriangles"  
        tau  = 1
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R ) 
    end
    println("Number of elements: ", mesh.nel, " number of dofs: ", mesh.nf)

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
    println("Model configuration :")
    @time SetUpProblem!(mesh, Pa, Vxa, Vya, Sxxa, Syya, Sxya, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, sex, sey)

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    @time ae, be, ze = ComputeFCFV(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, tau)

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    @time Kuu_v, fu_v, Kup_v, fp = ElementAssemblyLoop(mesh, ae, be, ze, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, tau)

    # Assemble triplets and sparse
    println("Assemble triplets and sparse:")
    @time Kuu, fu, Kup = CreateTripletsSparse(mesh, Kuu_v, fu_v, Kup_v)
    # display(UnicodePlots.spy(Kuu))
    # display(UnicodePlots.spy(Kup))

    # Solve for hybrid variable
    println("Linear solve:")
    @time Vxh, Vyh, Pe = StokesSolvers(mesh, Kuu, Kup, fu, fp, solver)

    # # Reconstruct element values
    println("Compute element values:")
    @time Vxe, Vye, Txxe, Tyye, Txye = ComputeElementValues(mesh, Vxh, Vyh, Pe, ae, be, ze, VxDir, VyDir, tau)

    # # Compute discretisation errors
    err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii = ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe )
    @printf("Error in Vx : %2.2e\n", err_Vx )
    @printf("Error in Vy : %2.2e\n", err_Vy )
    @printf("Error in Txx: %2.2e\n", err_Txx)
    @printf("Error in Tyy: %2.2e\n", err_Tyy)
    @printf("Error in Txy: %2.2e\n", err_Txy)
    @printf("Error in P  : %2.2e\n", err_P  )

    # # Visualise
    # println("Visualisation:")
    # # @time PlotMakie( mesh, sex )
    # @time PlotMakie( mesh, Pe )
    # # PlotElements( mesh )
    ndof = 2*mesh.nf+mesh.nel
    return ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii
end

N          = [8, 16, 32, 64, 128, 256]#, 512, 1024]
mesh_type  = "Quadrangles"
eV_quad    = zeros(size(N))
eP_quad    = zeros(size(N))
eTau_quad  = zeros(size(N))
t_quad     = zeros(size(N))
ndof_quad  = zeros(size(N))
for k=1:length(N)
    t_quad[k]    = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii = main( N[k], mesh_type )
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
    t_tri[k]     = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii  = main( N[k], mesh_type )
    eV_tri[k]    = err_V
    eP_tri[k]    = err_P
    eTau_tri[k]  = err_Tii
    ndof_tri[k]  = ndof
end

p = Plots.plot(  log10.(1.0 ./ N) , log10.(eV_quad), markershape=:rect,      label="Quads V"                          )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_quad), markershape=:rect,      label="Quads P"                          )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_quad), markershape=:rect,      label="Quads Tau"                        )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eV_tri),  markershape=:dtriangle, label="Triangles V"                     )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_tri),  markershape=:dtriangle, label="Triangles P"                     )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_tri),  markershape=:dtriangle, label="Triangles Tau", legend=:bottomright, xlabel = "log_10(h_x)", ylabel = "log_10(err_T)" )
# p = Plots.plot(  ndof_quad[2:end], t_quad[2:end], markershape=:rect,      label="Quads"                          )
# p = Plots.plot!( ndof_tri[2:end],  t_tri[2:end],  markershape=:dtriangle, label="Triangles", legend=:bottomright, xlabel = "ndof", ylabel = "time" )
display(p)