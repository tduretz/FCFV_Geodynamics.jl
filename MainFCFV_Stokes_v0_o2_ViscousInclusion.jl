import UnicodePlots, Plots
using  Revise, Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT

include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV_Stokes.jl")
include("SolversFCFV_Stokes.jl")
include("EvalAnalDani.jl")
include("DiscretisationFCFV_Stokes_o2.jl")
include("MarkerRoutines.jl") 

#--------------------------------------------------------------------#
    
function StabParam(τr, Γ, Ω, mesh_type, k)
    if mesh_type=="Quadrangles";        τ = k*τr/Γ end
    if mesh_type=="UnstructTriangles";  τ = τr   end 
    return τ
end

#--------------------------------------------------------------------#

function SetUpProblem!(mesh, P, P_f, Vx, Vy, Sxx, Syy, Sxy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, sx, sy, R,  η, gbar)
    # Evaluate T analytic on cell faces
     ηm,  ηi =  η[1],  η[2]
    for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        vx, vy, p, sxx, syy, sxy = EvalAnalDani( x, y, R,  ηm,  ηi )
        VxDir[in] = vx
        VyDir[in] = vy
        # Stress at faces - Pseudo-tractions
        p, dVxdx, dVxdy, dVydx, dVydy = Tractions( x, y, R,  ηm,  ηi, 1 )
        SxxNeu[in] = - p +  ηm*dVxdx 
        SyyNeu[in] = - p +  ηm*dVydy 
        SxyNeu[in] =        ηm*dVxdy
        SyxNeu[in] =        ηm*dVydx
        P_f[in]    = p
    end
    # Evaluate T analytic on barycentres
    for iel=1:mesh.nel
        x        = mesh.xc[iel]
        y        = mesh.yc[iel]
        vx, vy, pre, sxx, syy, sxy = EvalAnalDani( x, y, R,  ηm,  ηi )
        P[iel]   = pre
        Vx[iel]  = vx
        Vy[iel]  = vy
        Sxx[iel] = sxx
        Syy[iel] = syy
        Sxy[iel] = sxy
        sx[iel]  = 0.0
        sy[iel]  = 0.0
        out          = mesh.phase[iel] == 1.0
        mesh.ke[iel] = (out==1) * 1.0* ηm + (out!=1) * 1.0* ηi
        # Compute jump condition
        for ifac=1:mesh.nf_el
            # Face
            nodei  = mesh.e2f[iel,ifac]
            xF     = mesh.xf[nodei]
            yF     = mesh.yf[nodei]
            nodei  = mesh.e2f[iel,ifac]
            ni_x   = mesh.n_x[iel,ifac]
            ni_y   = mesh.n_y[iel,ifac]
            phase1 = Int64(mesh.phase[iel])
            pre1, dVxdx1, dVxdy1, dVydx1, dVydy1 = Tractions( xF, yF, R,  ηm,  ηi, phase1 )
             η_face1 =  η[phase1]
            tL_x  = ni_x*(-pre1 +  η_face1*dVxdx1) + ni_y* η_face1*dVxdy1
            tL_y  = ni_y*(-pre1 +  η_face1*dVydy1) + ni_x* η_face1*dVydx1
            # From element 2
            ineigh = (mesh.e2e[iel,ifac]>0) * mesh.e2e[iel,ifac] + (mesh.e2e[iel,ifac]<1) * iel
            phase2 = Int64(mesh.phase[ineigh])
            pre2, dVxdx2, dVxdy2, dVydx2, dVydy2 = Tractions( xF, yF, R,  ηm,  ηi, phase2)
             η_face2 =  η[phase2]
            tR_x = ni_x*(-pre2 +  η_face2*dVxdx2) + ni_y* η_face2*dVxdy2
            tR_y = ni_y*(-pre2 +  η_face2*dVydy2) + ni_x* η_face2*dVydx2
            gbar[iel,ifac,1] = 0.5 * (tL_x - tR_x)
            gbar[iel,ifac,2] = 0.5 * (tL_y - tR_y)
        end          
    end
    return
end

#--------------------------------------------------------------------#

function ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe, R,  η )
     ηm,  ηi =  η[1],  η[2]
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
    Pe_f = zeros(mesh.nf) 
   
    for in=1:mesh.nf

        P_f = 0
        cc  = 0
        if (mesh.f2e[in,1]>0)
            iel  = mesh.f2e[in,1]
            P_f += Pe[iel]
            cc  += 1
        end
        if (mesh.f2e[in,2]>0)
            iel  = mesh.f2e[in,2]
            P_f += Pe[iel]
            cc  += 1
        end
        P_f /= cc
        vx, vy, pre, sxx, syy, sxy = EvalAnalDani( mesh.xf[in], mesh.yf[in], R,  ηm,  ηi )
        Pe_f[in] = P_f - pre
    end
    for iel=1:mesh.nel
        x         = mesh.xc[iel]
        y         = mesh.yc[iel]
        vx, vy, pre, sxx, syy, sxy = EvalAnalDani( x, y, R,  ηm,  ηi )
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
    return errVx, errVy, errTxx, errTyy, errTxy, errP, Txxa, Tyya, Txya, Pe_f
end

#--------------------------------------------------------------------#

@views function main( n, mesh_type, τr, o2, new, markers, avg )
    # ```This version includes the jump condition derived from the analytical solution
    # The great thing is that pressure converges in L_infinity norm even with quadrangles - this rocks
    # # Test #1: For a contrast of:   η        = [10.0 1.0] (weak inclusion), tau = 20.0
    #     res = [1 2 4 8]*30 elements per dimension 
    #     err = [4.839 3.944 2.811 1.798]
    # # Test #2: For a contrast of:   η        = [1.0 100.0] (strong inclusion), tau = 1.0/5.0
    #     res = [1 2 4 8]*30 elements per dimension 
    #     err = [4.169 2.658 1.628 0.83]
    # ```

    println("\n******** FCFV STOKES ********")

    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    nx, ny     = 30*n, 30*n
    solver     = 1
    R          = 1.0
    inclusion  = 1
    η          = [1.0 100.0]
    # mesh_type  = "Quadrangles"
    # mesh_type  = "UnstructTriangles" 
    # mesh_type  = "TrianglesSameMATLAB"
    BC         = [2; 1; 1; 1] # S E N W --- 1: Dirichlet / 2: Neumann
    # new        = 1 # implementation if interface

    # Generate mesh
    if mesh_type=="Quadrangles" 
        tau  = 50.0   # for new = 0 leads to convergence 
        tau  = τr
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R, BC )
    elseif mesh_type=="UnstructTriangles" 
        tau = 1.0
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R, BC )
    elseif mesh_type=="TrianglesSameMATLAB"  
        tau  = 1.0#/5.0
        area = 1.0 # area factor: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        ninc = 29  # number of points that mesh the inclusion: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R, BC, area, ninc ) 
    end
    println("Number of elements: ", mesh.nel)

    Perr = 0

    # Source term and BCs etc...
    Pa     = zeros(mesh.nel)
    Pa_f   = zeros(mesh.nf)
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
    gbar   = zeros(mesh.nel,mesh.nf_el, 2)
    println("Model configuration :")
    @time SetUpProblem!(mesh, Pa, Pa_f, Vxa, Vya, Sxxa, Syya, Sxya, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, sex, sey, R,  η, gbar)

    if markers DoTheMarkerThing!( mesh, nx, ny, 4, 4, xmin, xmax, ymin, ymax, R,  η, avg ) end

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    FaceStabParam2( mesh, τr )
    println(minimum(mesh.τ))
    println(maximum(mesh.τ))
    if o2==0 @time ae, be, ze = ComputeFCFV(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu) end   
    if o2==1 @time ae, be, be_o2, ze, pe, mei, pe, rjx, rjy = ComputeFCFV_o2(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, o2) end

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    if o2==0 @time Kuu, Muu, Kup, fu, fp, tsparse = ElementAssemblyLoop(mesh, ae, be, ze, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, new) end
    if o2==1 @time Kuu, Muu, Kup, fu, fp, tsparse = ElementAssemblyLoop_o2(mesh, ae, be, be_o2, ze, mei, pe, rjx, rjy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, o2, new) end
    println("Sparsification: ", tsparse)

    # Solve for hybrid variable
    println("Linear solve:")
    @time Vxh, Vyh, Pe = StokesSolvers(mesh, Kuu, Kup, fu, fp, Muu, solver)

    # Reconstruct element values
    println("Compute element values:")
    if o2==0 @time Vxe, Vye, Txxe, Tyye, Txye = ComputeElementValues(mesh, Vxh, Vyh, Pe, ae, be, ze, VxDir, VyDir) end
    if o2==1 @time Vxe, Vye, Txxe, Tyye, Txye = ComputeElementValues_o2(mesh, Vxh, Vyh, Pe, ae, be, be_o2, ze, rjx, rjy, mei, VxDir, VyDir, o2) end

    # Compute discretisation errors
    err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, Txxa, Tyya, Txya, Perr_f = ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe, R,  η )
    @printf("Error in Vx : %2.2e\n", err_Vx )
    @printf("Error in Vy : %2.2e\n", err_Vy )
    @printf("Error in Txx: %2.2e\n", err_Txx)
    @printf("Error in Tyy: %2.2e\n", err_Tyy)
    @printf("Error in Txy: %2.2e\n", err_Txy)
    @printf("Error in P  : %2.2e\n", err_P  )

    Perr = abs.(Pa.-Pe) 
    Verr = sqrt.( (Vxe.-Vxa).^2 .+ (Vye.-Vya).^2 ) 
    println("L_inf P error: ", maximum(Perr), " --- L_inf V error: ", maximum(Verr))
    println("L_inf P error: ", maximum(Perr_f), " --- L_inf V error: ", maximum(Verr))

    # Perr_fc= zeros(mesh.nel) 
    # for e=1:mesh.nel
    #     for i=1:mesh.nf_el
    #         nodei = mesh.e2f[e,i]
    #         Perr_fc[e] += 0.25*Perr_f[nodei]
    #     end
    # end

    # Visualise
    # println("Visualisation:")
    # PlotMakie(mesh, v, xmin, xmax, ymin, ymax; cmap = :viridis, min_v = minimum(v), max_v = maximum(v))
    # @time PlotMakie( mesh, Verr, xmin, xmax, ymin, ymax, :jet1, minimum(Verr), maximum(Verr) )
    @time PlotMakie( mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1, min_v =minimum(Pa), max_v =maximum(Pa) )
    # @time PlotMakie( mesh, Perr, xmin, xmax, ymin, ymax, :jet1, minimum(Perr), maximum(Perr) )
    # @time PlotMakie( mesh, Txxe, xmin, xmax, ymin, ymax, :jet1, -6.0, 2.0 )
    # @time PlotMakie( mesh, mesh.ke, xmin, xmax, ymin, ymax; cmap=:jet1 )
    # @time PlotMakie( mesh, mesh.phase, xmin, xmax, ymin, ymax, :jet1)

    return maximum(Perr), maximum(Perr_f)
end

#--------------------------------------------------------------------#

N       = 1:16
new     = 0
n       = 8
τ       = 50.0
o2      = 0
markers = false
avg     = 0

eP_quad_o1_Linf_c = zeros(length(N))
eP_quad_o1_Linf_f = zeros(length(N))

for i=1:length(N)
    res = main( N[i], "Quadrangles", 1/4.0, o2, new, markers, avg )
    eP_quad_o1_Linf_c[i] = res[1]
    eP_quad_o1_Linf_f[i] = res[2]
    # main( n, "UnstructTriangles", 0.1, o2, new, markers, avg )

    p = Plots.plot(  log10.(1.0 ./ N[1:i]) , log10.(eP_quad_o1_Linf_c[1:i]),   markershape=:rect,      color=:red,      linestyle = :dot,  label="Quads P O1 c"                          )
    p = Plots.plot!( log10.(1.0 ./ N[1:i]) , log10.(eP_quad_o1_Linf_f[1:i]),   markershape=:rect,      color=:blue,      linestyle = :dot,  label="Quads P O1 f"                          )
    p = Plots.plot!( legend=:outertopright, xlabel = "log_10(h_x)", ylabel = "log_10(err_T)" )
    display(p)
end

#--------------------------------------------------------------------#

# τ    = 1:5:200
# perr = zeros(size(τ))
# for i=1:length(τ)
#     perr[i] = main(τ[i])
# end
# display(Plots.plot(τ, perr, title=minimum(perr))

# main( 1, "Quadrangles", 1.0, o2, new, markers, avg )