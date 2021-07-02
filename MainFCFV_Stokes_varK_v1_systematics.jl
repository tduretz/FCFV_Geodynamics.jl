import UnicodePlots, Plots
using  Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT

include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV_Stokes.jl")
include("SolversFCFV_Stokes.jl")
include("EvalAnalDani_v2.jl")

#--------------------------------------------------------------------#

function SetUpProblem!(mesh, P, Vx, Vy, Sxx, Syy, Sxy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, sx, sy, rc, eta, gbar)
    # Evaluate T analytic on cell faces
    etam, etac = eta[1], eta[2]
    gr  = 0;                        # Simple shear: gr=1, er=0
    er  = -1;                       # Strain rate
    @tturbo for ifac=1:mesh.nf
        x, y                           = mesh.xf[ifac], mesh.yf[ifac]
        in                             = sqrt(x^2.0 + y^2.0)<=rc
        pa                             = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
        vxa, vya                       = SolutionFields_v( etam, etac, rc, er, gr, x, y, in)
        VxDir[ifac] = vxa
        VyDir[ifac] = vya
        # Pseudo-tractions
        dvxdxa, dvxdya, dvydxa, dvydya = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
        SxxNeu[ifac] = - pa +  etam*dvxdxa 
        SyyNeu[ifac] = - pa +  etam*dvydya
        SxyNeu[ifac] =         etam*dvxdya
        SyxNeu[ifac] =         etam*dvydxa
    end
    # Evaluate T analytic on barycentres
    @tturbo for iel=1:mesh.nel
        x, y                           = mesh.xc[iel], mesh.yc[iel]
        in                             = sqrt(x^2.0 + y^2.0)<=rc
        pa                             = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
        vxa, vya                       = SolutionFields_v( etam, etac, rc, er, gr, x, y, in)
        dvxdxa, dvxdya, dvydxa, dvydya = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
        P[iel]       = pa
        Vx[iel]      = vxa
        Vy[iel]      = vya
        etaa         = (in==0) * etam + (in==1) * etac
        Sxx[iel]     = -pa + 2.0*etaa*dvxdxa
        Syy[iel]     = -pa + 2.0*etaa*dvydya
        Sxy[iel]     = etaa*(dvxdya+dvydxa)
        sx[iel]      = 0.0
        sy[iel]      = 0.0
        out          = mesh.phase[iel] == 1.0
        mesh.ke[iel] = (out==1) * 1.0*etam + (out!=1) * 1.0*etac
    end

    # Compute jump condition
    @tturbo for iel=1:mesh.nel
        x, y                           = mesh.xc[iel], mesh.yc[iel] 
        for ifac=1:mesh.nf_el
            # Face
            nodei  = mesh.e2f[iel,ifac]
            xF     = mesh.xf[nodei]
            yF     = mesh.yf[nodei]
            nodei  = mesh.e2f[iel,ifac]
            bc     = mesh.bc[nodei]
            dAi    = mesh.dA[iel,ifac]
            ni_x   = mesh.n_x[iel,ifac]
            ni_y   = mesh.n_y[iel,ifac]
            phase1 = Int64(mesh.phase[iel])
            in     = (phase1==1) * 0 + (phase1==2) * 1 
            pre1                           = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
            dVxdx1, dVxdy1, dVydx1, dVydy1 = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
            eta_face1 = (in==0) * etam + (in==1) * etac
            tL_x  = ni_x*(-pre1 + eta_face1*dVxdx1) + ni_y*eta_face1*dVxdy1
            tL_y  = ni_y*(-pre1 + eta_face1*dVydy1) + ni_x*eta_face1*dVydx1
            # From element 2
            ineigh = (mesh.e2e[iel,ifac]>0) * mesh.e2e[iel,ifac] + (mesh.e2e[iel,ifac]<1) * iel
            phase2 = Int64(mesh.phase[ineigh])
            in     = (phase2==1) * 0 + (phase2==2) * 1 
            pre2                           = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
            dVxdx2, dVxdy2, dVydx2, dVydy2 = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
            eta_face2 = (in==0) * etam + (in==1) * etac
            tR_x = ni_x*(-pre2 + eta_face2*dVxdx2) + ni_y*eta_face2*dVxdy2
            tR_y = ni_y*(-pre2 + eta_face2*dVydy2) + ni_x*eta_face2*dVydx2
            gbar[iel,ifac,1] = 0.5 * (tL_x - tR_x)
            gbar[iel,ifac,2] = 0.5 * (tL_y - tR_y)
        end          
    end
    return
end

#--------------------------------------------------------------------#

function ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe, rc, eta )
    etam, etac = eta[1], eta[2]
    gr  = 0;                        # Simple shear: gr=1, er=0
    er  = -1;                       # Strain rate
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
    @tturbo for iel=1:mesh.nel
        x, y                           = mesh.xc[iel], mesh.yc[iel] 
        in                             = sqrt(x^2.0 + y^2.0)<=rc
        pa                             = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
        vxa, vya                       = SolutionFields_v( etam, etac, rc, er, gr, x, y, in)
        dvxdxa, dvxdya, dvydxa, dvydya = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
        Pa[iel], Vxa[iel], Vya[iel]    = pa, vxa, vya
        Txxa[iel] = 2.0*mesh.ke[iel]*dvxdxa
        Tyya[iel] = 2.0*mesh.ke[iel]*dvydya 
        Txya[iel] = 1.0*mesh.ke[iel]*(dvxdya + dvydxa) 
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
    
function StabParam(tau, dA, Vol, mesh_type, coeff)
    if mesh_type=="Quadrangles";        taui = coeff*tau  end
    if mesh_type=="UnstructTriangles";  taui = coeff*tau*dA  end
    return taui
end

#--------------------------------------------------------------------#

@views function main(n, tau)

    println("\n******** FCFV STOKES ********")

    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    # n          = 1
    nx, ny     = 30*n, 30*n
    solver     = 1
    R          = 1.0
    inclusion  = 1
    eta        = [1.0 100.0]
    mesh_type  = "Quadrangles"
    # mesh_type  = "UnstructTriangles"
    BC         = [2; 1; 1; 1] # S E N W --- 1: Dirichlet / 2: Neumann

    # Generate mesh
    if mesh_type=="Quadrangles" 
        # tau  = 2.0   
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC )
    elseif mesh_type=="UnstructTriangles"  
        tau  = 1.0/5.0
        area = 1.0 # area factor: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        ninc = 29  # number of points that mesh the inclusion: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC, area, ninc ) 
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
    @time SetUpProblem!(mesh, Pa, Vxa, Vya, Sxxa, Syya, Sxya, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, sex, sey, R, eta, gbar)

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    @time ae, be, ze = ComputeFCFV(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, tau)

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    @time Kuu_v, fu_v, Kup_v, fp = ElementAssemblyLoop(mesh, ae, be, ze, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, tau)

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

    Perr = abs.(Pa.-Pe) 
    Verr = sqrt.( (Vxe.-Vxa).^2 .+ (Vye.-Vya).^2 ) 
    println("L_inf P error: ", maximum(Perr), " --- L_inf V error: ", maximum(Verr))

    # # Visualise
    # println("Visualisation:")
    # # PlotMakie(mesh, v, xmin, xmax, ymin, ymax; cmap = :viridis, min_v = minimum(v), max_v = maximum(v))
    # # @time PlotMakie( mesh, Verr, xmin, xmax, ymin, ymax, :jet1, minimum(Verr), maximum(Verr) )
    # @time PlotMakie( mesh, Pe, xmin, xmax, ymin, ymax, :jet1, minimum(Pa), maximum(Pa) )
    # # @time PlotMakie( mesh, Perr, xmin, xmax, ymin, ymax, :jet1, minimum(Perr), maximum(Perr) )
    # # @time PlotMakie( mesh, Txxe, xmin, xmax, ymin, ymax, :jet1, -6.0, 2.0 )
    # # @time PlotMakie( mesh, (mesh.ke), xmin, xmax, ymin, ymax, :jet1 )
    # # @time PlotMakie( mesh, mesh.phase, xmin, xmax, ymin, ymax, :jet1)

    return maximum(Perr), maximum(Verr)
end

# n = 1
# tau = 1.0
# main(n, tau)

# n    = collect(1:1:16)
# n    = collect(17:1:20) # p2
# tau  = collect(4:1:25)
n    = collect(1:1:10)
tau  = collect(1:1:4)
resu = zeros(length(n), length(tau))
resp = zeros(length(n), length(tau))

for in = 1:length(n)
    for it = 1:length(tau)
        rp, ru = main(n[in], tau[it])
        resu[in,it] = ru
        resp[in,it] = rp
    end
end

# p2 = Plots.heatmap(n, tau, resp', c=:jet1 )
# display(Plots.plot(p2))

file = matopen(string(@__DIR__,"/results/MaxPerr_p3.mat"), "w" )
write(file, "n",        n )
write(file, "tau",    tau )
write(file, "resu",  resu )
write(file, "resp",  resp )
close(file)