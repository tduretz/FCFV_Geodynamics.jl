import UnicodePlots, Plots
using  Revise, Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT, Base.Threads

include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV_Stokes.jl")
include("SolversFCFV_Stokes.jl")
include("EvalAnalDani_v2.jl")
include("DiscretisationFCFV_Stokes_o2.jl")
BLAS.set_num_threads(1)
include("MarkerRoutines.jl") 

#--------------------------------------------------------------------#
    
function StabParam(τr, dA, Ω, mesh_type, coeff)
    if mesh_type=="Quadrangles";        τi = 1.0/coeff*τr*dA  end
    if mesh_type=="UnstructTriangles";  τi = coeff*τr*dA  end
    return τi
end

#--------------------------------------------------------------------#

function SetMarkers!( p, R )
    @tturbo for k=1:p.nmark
        in         = (p.x[k]^2 + p.y[k]^2) < R^2 
        p.phase[k] = (in==0)* 1.0 + (in==1)*2.0
    end
end

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
            dAi    = mesh.Γ[iel,ifac]
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

@views function main(n, mesh_type, τr, o2, new)

    println("\n******** FCFV STOKES ********")

    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    nx, ny     = 30*n, 30*n
    solver     = 1
    R          = 1.0
    inclusion  = 1
    eta        = [1.0 100.0]
    BC         = [2; 1; 1; 1] # S E N W --- 1: Dirichlet / 2: Neumann
    nmx        = 4            # 2 marker per cell in x
    nmy        = 4            # 2 marker per cell in y
    # Generate mesh
    if mesh_type=="Quadrangles" 
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R, BC )
    elseif mesh_type=="UnstructTriangles"  
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R, BC )
        # area = 1.0 # area factor: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        # ninc = 29  # number of points that mesh the inclusion: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        # mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC, area, ninc ) 
    end
    println("Number of elements: ", mesh.nel, " --- Number of dofs: ", 2*mesh.nf+mesh.nel, " --- tau0 =  ", τr )

    # Initialise markers
    ncx, ncy  = nx, ny
    nmark0    = ncx*ncy*nmx*nmy; # total initial number of marker in grid
    dx,  dy   = (xmax-xmin)/ncx, (ymax-ymin)/ncy
    dxm, dym  = dx/nmx, dy/nmy 
    xm1d      =  LinRange(xmin+dxm/2, xmax-dxm/2, ncx*nmx)
    ym1d      =  LinRange(ymin+dym/2, ymax-dym/2, ncy*nmy)
    (xmi,ymi) = ([x for x=xm1d,y=ym1d], [y for x=xm1d,y=ym1d])
    xc        =  LinRange(xmin+dx/2, xmax-dx/2, ncx)
    yc        =  LinRange(ymin+dy/2, ymax-dy/2, ncy)
    # Over allocate markers
    nmark_max = 1*nmark0;
    phm    = zeros(Float64, nmark_max)
    xm     = zeros(Float64, nmark_max) 
    ym     = zeros(Float64, nmark_max)
    cellxm = zeros(Int64,   nmark_max)
    cellym = zeros(Int64,   nmark_max)
    xm[1:nmark0]     = vec(xmi)
    ym[1:nmark0]     = vec(ymi)
    phm[1:nmark0]    = zeros(Float64, size(xmi))
    cellxm[1:nmark0] = zeros(Int64,   size(xmi)) #zeros(CartesianIndex{2}, size(xm))
    cellym[1:nmark0] = zeros(Int64,   size(xmi))
    p      = Markers( xm, ym, phm, cellxm, cellym, nmark0, nmark_max )

    # Define phase
    SetMarkers!( p, R )

    # Update cell info on markers
    LocateMarkers(p,dx,dy,xc,yc,xmin,xmax,ymin,ymax)

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

    k2d = ones(ncx,ncy)
    @time Markers2Cells2(p,k2d,xc,yc,dx,dy,ncx,ncy,eta,1)
    if mesh_type=="Quadrangles" 
        mesh.ke .= k2d[:]
    end

    # Need to compute it after initialising viscosity field using markers
    FaceStabParam( mesh, τr, mesh_type )

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    @time ae, be, be_o2, ze, pe, mei, pe, rjx, rjy = ComputeFCFV_o2(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, o2)

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    # @time Kuu_v, fu_v, Kup_v, fp = ElementAssemblyLoop(mesh, ae, be, ze, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, tau)
    @time Kuu, Muu, Kup, fu, fp, tsparse = ElementAssemblyLoop_o2(mesh, ae, be, be_o2, ze, mei, pe, rjx, rjy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, o2, new)

    # Solve for hybrid variable
    println("Linear solve:")
    @time Vxh, Vyh, Pe = StokesSolvers(mesh, Kuu, Kup, fu, fp, Muu, solver)

    # # Reconstruct element values
    println("Compute element values:")
    @time Vxe, Vye, Txxe, Tyye, Txye = ComputeElementValues_o2(mesh, Vxh, Vyh, Pe, ae, be, be_o2, ze, rjx, rjy, mei, VxDir, VyDir, o2)

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

    # Visualise
    println("Visualisation:")
    # @time PlotMakie( mesh, Vxe, xmin, xmax, ymin, ymax; cmap=:jet1,  min_v=minimum(Vxa),  max_v=maximum(Vxa), writefig=false)
    # @time PlotMakie( mesh, Verr, xmin, xmax, ymin, ymax, :jet1, minimum(Verr), maximum(Verr) )
    # @time PlotMakie( mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1,  min_v=minimum(Pa),  max_v=maximum(Pa), writefig=false )
    # @time PlotMakie( mesh, Perr, xmin, xmax, ymin, ymax, :jet1, minimum(Perr), maximum(Perr) )
    # @time PlotMakie( mesh, Txxe, xmin, xmax, ymin, ymax, :jet1, -6.0, 2.0 )
    @time PlotMakie( mesh, (mesh.ke), xmin, xmax, ymin, ymax; cmap=:jet1 )
    # @time PlotMakie( mesh, mesh.phase, xmin, xmax, ymin, ymax; cmap=:jet1 )

    return maximum(Perr), maximum(Verr)
end

new = 0 
n   = 1
τ   = 1.0
o2  = 0
# main(2, "Quadrangles", 25, 1) # L_INF P 1.67 no-interp
# main(2, "Quadrangles", 50, 1) # L_INF P 1.18 arith
# main(4, "Quadrangles", 180, 1) # L_INF P 1.45 arith
# main(8, "Quadrangles", 280, 1) # L_INF P 1.8 arith
# main(2, "Quadrangles", 25, 1)
# main(n, "UnstructTriangles", τ, o2, new)
main(n, "Quadrangles", 10000, o2, new)