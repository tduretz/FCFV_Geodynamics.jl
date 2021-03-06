import UnicodePlots, Plots
using  Revise, Printf, LoopVectorization, LinearAlgebra, SparseArrays

include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV_Stokes.jl")
include("SolversFCFV_Stokes.jl")
include("EvalAnalDani.jl")

#--------------------------------------------------------------------#

function SetUpProblem!(mesh, P, Vx, Vy, Sxx, Syy, Sxy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, sx, sy, R, eta, gbar)
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
        # Pseudo-tractions
        pre, dVxdx, dVxdy, dVydx, dVydy = Tractions( x, y, R, etam, etai, 1 )
        SxxNeu[in] = - pre + etam*dVxdx 
        SyyNeu[in] = - pre + etam*dVydy 
        SxyNeu[in] =         etam*dVxdy
        SyxNeu[in] =         etam*dVydx
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
        # Compute jump condition
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
            pre1, dVxdx1, dVxdy1, dVydx1, dVydy1 = Tractions( xF, yF, R, etam, etai, phase1 )
            eta_face1 = eta[phase1]
            tL_x  = ni_x*(-pre1 + eta_face1*dVxdx1) + ni_y*eta_face1*dVxdy1
            tL_y  = ni_y*(-pre1 + eta_face1*dVydy1) + ni_x*eta_face1*dVydx1
            # From element 2
            ineigh = (mesh.e2e[iel,ifac]>0) * mesh.e2e[iel,ifac] + (mesh.e2e[iel,ifac]<1) * iel
            phase2 = Int64(mesh.phase[ineigh])
            pre2, dVxdx2, dVxdy2, dVydx2, dVydy2 = Tractions( xF, yF, R, etam, etai, phase2)
            eta_face2 = eta[phase2]
            tR_x = ni_x*(-pre2 + eta_face2*dVxdx2) + ni_y*eta_face2*dVxdy2
            tR_y = ni_y*(-pre2 + eta_face2*dVydy2) + ni_x*eta_face2*dVydx2
            gbar[iel,ifac,1] = 0.5 * (tL_x - tR_x)
            gbar[iel,ifac,2] = 0.5 * (tL_y - tR_y)
        end          
    end

    # # detect cells at interface and set to geometric mean
    # for iel=1:mesh.nel

    #     for ifac=1:mesh.nf_el
    #         nodei  = mesh.e2f[iel,ifac]
    #         if mesh.bc[nodei] == 3
    #             # mesh.ke[iel] = (etam*etai)^(1/2)
    #             if Int64(mesh.phase[iel]) == 1
    #                 mesh.ke[iel] = 2.0/(1.0/etam+1.0/etai)
    #             else
    #                 mesh.ke[iel] = 0.5*(etam+etai)
    #             end
    #         end
    #     end
    # end


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
        pre, dVxdx, dVxdy, dVydx, dVydy = Tractions( x, y, R, etam, etai, mesh.phase[iel] )
        Pa[iel]   = pre
        Vxa[iel]  = vx
        Vya[iel]  = vy
        Txxa[iel] = 2.0*mesh.ke[iel]*dVxdx
        Tyya[iel] = 2.0*mesh.ke[iel]*dVydy 
        Txya[iel] = 1.0*mesh.ke[iel]*(dVxdy+dVydx) 
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

@views function main( n, solver)
    # ```This version include the jump condition derived from the analytical solution
    # The great thing is that pressure converges in L_infinity norm evem with quadrangles - this rocks
    # # Test #1: For a contrast of:  eta        = [10.0 1.0] (weak inclusion), tau = 20.0
    #     res = [1 2 4 8]*30 elements per dimension 
    #     err = [4.839 3.944 2.811 1.798]
    # # Test #2: For a contrast of:  eta        = [1.0 100.0] (strong inclusion), tau = 1.0/5.0
    #     res = [1 2 4 8]*30 elements per dimension 
    #     err = [4.169 2.658 1.628 0.83]
    # ```

    println("\n\n******** FCFV STOKES ********")

    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    # n          = 16
    nx, ny     = 30*n, 30*n
    # solver     = 3
    R          = 1.0
    inclusion  = 1
    eta        = [1.0 100.0]
    # mesh_type  = "Quadrangles"
    mesh_type  = "UnstructTriangles"
    BC         = [1; 1; 1; 1] # S E N W --- 1: Dirichlet / 2: Neumann

    # Generate mesh
    if mesh_type=="Quadrangles" 
        tau  = 4.0   
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC )
    elseif mesh_type=="UnstructTriangles"  
        tau  = 1.0/5.0
        # area = 1.0 # area factor: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        # ninc = 29  # number of points that mesh the inclusion: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC ) 
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
    t_solve = @elapsed  Vxh, Vyh, Pe = StokesSolvers(mesh, Kuu, Kup, fu, fp, solver)
    println("Linear solver took: ", t_solve, " s")

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
    # # @time PlotMakie(mesh, Vxe, xmin, xmax, ymin, ymax, :jet1)
    # # @time PlotMakie( mesh, Verr, xmin, xmax, ymin, ymax, :jet1, minimum(Verr), maximum(Verr) )
    # # @time PlotMakie( mesh, Pe, xmin, xmax, ymin, ymax, :jet1, minimum(Pa), maximum(Pa) )
    # # @time PlotMakie( mesh, Perr, xmin, xmax, ymin, ymax, :jet1, minimum(Perr), maximum(Perr) )
    # # @time PlotMakie( mesh, Txxe, xmin, xmax, ymin, ymax, :jet1, -6.0, 2.0 )
    @time PlotMakie( mesh, Tyya, xmin, xmax, ymin, ymax, :jet1, -2.0, 6.0 )
    # # @time PlotMakie( mesh, log10.(mesh.ke), xmin, xmax, ymin, ymax, :jet1 )
    # # @time PlotMakie( mesh, mesh.phase, xmin, xmax, ymin, ymax, :jet1)

    ndof = 2*mesh.nf+mesh.nel
    print(t_solve)
    return t_solve, ndof

end

N = 1:1
ndofs = zeros(length(N) )
t0    = zeros(length(N) )
t1    = zeros(length(N) )
t3    = zeros(length(N) )

main( 1, 1 ) # warm up

for i=1:length(N)
    # t0[i], ndofs[i] = main( N[i], 0 )
    # t1[i], ndofs[i]          = main( N[i], 1 )
    t3[i] , ndofs[i]          = main( N[i], 3 )
end

# p = Plots.plot(  ndofs,  t0,  markershape=:dtriangle, label="Monolithic", legend=:bottomright, xlabel = "ndof", ylabel = "solve time" )
# p = Plots.plot( ndofs,  t1,  markershape=:dtriangle, label="Powell-Hestenes", legend=:bottomright, xlabel = "ndof", ylabel = "solve time" )
p = Plots.plot( ndofs,  t3,  markershape=:dtriangle, label="Modified PH", legend=:bottomright, xlabel = "ndof", ylabel = "solve time" )
display(p)