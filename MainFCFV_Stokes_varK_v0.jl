import TriangleMesh, UnicodePlots, Plots
using Printf, LoopVectorization, LinearAlgebra, SparseArrays

include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV_Stokes.jl")
include("SolversFCFV_Stokes.jl")
include("EvalAnalDani.jl")

#--------------------------------------------------------------------#

function SetUpProblem!(mesh, P, Vx, Vy, Sxx, Syy, Sxy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, sx, sy, R, eta)
    # Evaluate T analytic on cell faces
    etam, etai = eta[1], eta[2]
    for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        vx, vy, pre, sxx, syy, sxy = EvalAnalDani( x, y, R, etam, etai )
        VxDir[in] = vx
        VyDir[in] = vy
        # Stress at faces
        p          = pre
        SxxNeu[in] = sxx
        SyyNeu[in] = syy
        SxyNeu[in] = sxy
    end
    # Evaluate T analytic on barycentres
    for iel=1:mesh.nel
        x        = mesh.xc[iel]
        y        = mesh.yc[iel]
        vx, vy, pre, sxx, syy, sxy = EvalAnalDani( x, y, R, etam, etai )
        P[iel]   = pre
        Vx[iel]  = vx
        Vy[iel]  = vy
        Sxx[iel] = sxx
        Syy[iel] = syy
        Sxy[iel] = sxy
        sx[iel]  = 0.0
        sy[iel]  = 0.0
        out          = mesh.phase[iel] == 1.0
        mesh.ke[iel] = (out==1) * 1.0*etam + (out!=1) * 1.0*etai
    end
    return
end

#--------------------------------------------------------------------#

function ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe, R, eta )
    etam, etai = eta[1], eta[2]
    eVx  = zeros(mesh.nel)
    eVy  = zeros(mesh.nel)
    eTxx = zeros(mesh.nel)
    eTyy = zeros(mesh.nel)
    eTxy = zeros(mesh.nel)
    eP   = zeros(mesh.nel)
    Vxa  = zeros(mesh.nel)
    Vya  = zeros(mesh.nel)
    Txxa = zeros(mesh.nel)
    Tyya = zeros(mesh.nel)
    Txya = zeros(mesh.nel)
    Pa   = zeros(mesh.nel)
    for iel=1:mesh.nel
        x         = mesh.xc[iel]
        y         = mesh.yc[iel]
        vx, vy, pre, sxx, syy, sxy = EvalAnalDani( x, y, R, etam, etai )
        Pa[iel]   = pre
        Vxa[iel]  = vx
        Vya[iel]  = vy
        Txxa[iel] = pre + sxx 
        Tyya[iel] = pre + syy 
        Txya[iel] = sxy
        eVx[iel]  = Vxe[iel]  - Vxa[iel]
        eVy[iel]  = Vye[iel]  - Vya[iel]
        eTxx[iel] = Txxe[iel] - Txxa[iel]
        eTyy[iel] = Tyye[iel] - Tyya[iel]
        eTxy[iel] = Txye[iel] - Txya[iel]
        eP[iel]   = Pe[iel]   - Pa[iel]
    end
    errVx  = norm(eVx) /norm(Vxa)
    errVy  = norm(eVy) /norm(Vya)
    errTxx = norm(eTxx)/norm(Txxa)
    errTyy = norm(eTyy)/norm(Tyya)
    errTxy = norm(eTxy)/norm(Txya)
    errP   = norm(eP)  /norm(Pa)
    return errVx, errVy, errTxx, errTyy, errTxy, errP, Txxa, Tyya, Txya
end

#--------------------------------------------------------------------#
    
function StabParam(tau, dA, Vol, mesh_type)
    if mesh_type=="Quadrangles";        taui = tau  end
    if mesh_type=="UnstructTriangles";  taui = tau  end
    return taui
end

#--------------------------------------------------------------------#

@views function main()

    println("\n******** FCFV STOKES ********")

    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    n          = 4
    nx, ny     = 30*n, 30*n
    solver     = 0
    R          = 1.0
    inclusion  = 1
    eta        = [1.0 100.0]
    mesh_type  = "Quadrangles"
    # mesh_type  = "UnstructTriangles"
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        tau  = 5
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R )
    elseif mesh_type=="UnstructTriangles"  
        tau  = 5
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R ) 
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
    println("Model configuration :")
    @time SetUpProblem!(mesh, Pa, Vxa, Vya, Sxxa, Syya, Sxya, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, sex, sey, R, eta)

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
    err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, Txxa, Tyya, Txya = ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe, R, eta )
    @printf("Error in Vx : %2.2e\n", err_Vx )
    @printf("Error in Vy : %2.2e\n", err_Vy )
    @printf("Error in Txx: %2.2e\n", err_Txx)
    @printf("Error in Tyy: %2.2e\n", err_Tyy)
    @printf("Error in Txy: %2.2e\n", err_Txy)
    @printf("Error in P  : %2.2e\n", err_P  )

    println(minimum(Txya))
    println(minimum(Txye))
    println(maximum(Txya))
    println(maximum(Txye))

    # Visualise
    println("Visualisation:")
    @time PlotMakie( mesh, Txye, xmin, xmax, ymin, ymax, :jet1, [-0.5 0.5] )

end

main()