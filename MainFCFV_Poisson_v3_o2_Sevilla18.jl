import UnicodePlots
using Printf, LoopVectorization, LinearAlgebra, SparseArrays

include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV.jl")
include("DiscretisationFCFV_Poisson_o2.jl")
include("SolversFCFV_Poisson.jl")

#--------------------------------------------------------------------#

function SetUpProblem!( mesh, T, Tdir, Tneu, se, a, b, c, d, alp, bet )
    # Evaluate T analytic on cell faces
    @tturbo for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        Tdir[in] = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        dTdx     = Tdir[in] * (a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y))
        dTdy     = Tdir[in] * (alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y))
        Tneu[in] = -dTdy # nx*dTdx + nt*dTdy on SOUTH face
    end
    # Evaluate T analytic on barycentres
    @tturbo for iel=1:mesh.nel
        x       = mesh.xc[iel]
        y       = mesh.yc[iel]
        T       = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        T[iel]  = T
        se[iel] = T*(-a*alp*cos(a*x + c*y) + b*bet*sin(b*x + d*y))*(a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y)) + T*(a^2*alp*sin(a*x + c*y) + b^2*bet*cos(b*x + d*y)) + T*(-alp*c*cos(a*x + c*y) + bet*d*sin(b*x + d*y))*(alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y)) + T*(alp*c^2*sin(a*x + c*y) + bet*d^2*cos(b*x + d*y))
    end
    return
end

#--------------------------------------------------------------------#

function ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    eT  = zeros(mesh.nel)
    eqx = zeros(mesh.nel)
    eqy = zeros(mesh.nel)
    eq  = zeros(mesh.nel)
    Ta  = zeros(mesh.nel)
    qxa = zeros(mesh.nel)
    qya = zeros(mesh.nel)
    qa  = zeros(mesh.nel)
    @tturbo for iel=1:mesh.nel
        x        = mesh.xc[iel]
        y        = mesh.yc[iel]
        Ta[iel]  = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        qxa[iel] = -Ta[iel] * (a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y))
        qya[iel] = -Ta[iel] * (alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y))
        qa[iel]  = sqrt(qxa[iel]^2 + qya[iel]^2)
        q        = sqrt( qx[iel]^2 +  qy[iel]^2)
        eT[iel]  = Te[iel] - Ta[iel]
        eqx[iel] = qx[iel] - qxa[iel]
        eqy[iel] = qy[iel] - qya[iel]
        eq[iel]  = q       - qa[iel]
    end
    errT  = norm(eT) /norm(Ta)
    errqx = norm(eqx)/norm(qxa)
    errqy = norm(eqy)/norm(qya)
    errq  = norm(eq) /norm(qa)
    return errT, errqx, errqy, errq
end
    
#--------------------------------------------------------------------#

function StabParam(tau, dA, Vol, mesh_type, coeff)
    if mesh_type=="Quadrangles";        taui = tau  end
    if mesh_type=="UnstructTriangles";  taui = tau  end
    return taui
end

#--------------------------------------------------------------------#
    
@views function main( N, mesh_type, order )

    println("\n******** FCFV POISSON ********")

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = N, N
    R          = 0.5
    inclusion  = 0
    solver     = 0
    o2         = order-1
    BC         = [1; 1; 1; 1] # S E N W --- 1: Dirichlet / 2: Neumann
    # mesh_type  = "Quadrangles"
    # mesh_type  = "UnstructTriangles"
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        if o2==0 tau  = 1   end # 1.0 ---> reproduces figure from paper
        if o2==1 tau  = 1e3 end
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, tau, inclusion, R, BC )
    elseif mesh_type=="UnstructTriangles"  
        if o2==0 tau  = 1e0 end
        if o2==1 tau  = 1e3 end
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, tau, inclusion, R, BC ) 
    end
    println("Number of elements: ", mesh.nel)

    # Source term and BCs etc...
    Tanal  = zeros(mesh.nel)
    se     = zeros(mesh.nel)
    Te     = zeros(mesh.nel)
    qx     = zeros(mesh.nel)
    qy     = zeros(mesh.nel)
    Tdir   = zeros(mesh.nf)
    Tneu   = zeros(mesh.nf)
    Rh     = zeros(mesh.nf)
    Th     = zeros(mesh.nf)
    alp = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;
    println("Model configuration :")
    @time SetUpProblem!(mesh , Tanal, Tdir, Tneu, se, a, b, c, d, alp, bet)

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    @time ae, be, be_o2, ze, pe, mei, pe, rj  = ComputeFCFV_o2(mesh, se, Tdir, tau, o2)

    # # Residual
    # Th[mesh.bc.==1] .= Tdir[mesh.bc.==1] 
    # @time Te, qx, qy = ComputeElementValues_o2(mesh, Th, ae, be, be_o2, ze, rj, mei, Tdir, tau, o2)

    # PoissonResidual(Rh, mesh, Th, Te, qx, qy)
    # @info norm(Th)
    # display(reshape(Te,8,8)')
    # display(reshape(qx,8,8)')
    # display(reshape(qy,8,8)')

    # Rxc = zeros(mesh.nel)
    # Ryc = zeros(mesh.nel)
    # # Compute residual of global equation
    # for iel=1:mesh.nel  
    #     Rxc[iel] = 0.5*(Rh[mesh.e2f[iel,1]] + Rh[mesh.e2f[iel,4]]) 
    #     Ryc[iel] = 0.5*(Rh[mesh.e2f[iel,2]] + Rh[mesh.e2f[iel,3]])   
    # end
    # display(reshape(Rxc,8,8)')
    # display(reshape(Ryc,8,8)')

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    @time Kv, fv = ElementAssemblyLoop_o2(mesh, ae, be, be_o2, ze, mei, pe, rj, Tdir, Tneu, tau, o2)

    # Assemble triplets and sparse
    println("Assemble triplets and sparse:")
    @time K, f = CreateTripletsSparse(mesh, Kv, fv)
    
    # Solve
    println("Solve:")
    @time Th = SolvePoisson(mesh, K, f, solver)

    # Reconstruct element values
    println("Compute element values:")
    @time Te, qx, qy = ComputeElementValues_o2(mesh, Th, ae, be, be_o2, ze, rj, mei, Tdir, tau, o2)

    # Residual
    PoissonResidual(Rh, mesh, Th, Te, qx, qy)
    @info norm(Rh)

    # Compute discretisation errors
    err_T, err_qx, err_qy, err_q = ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    println("Error in T:  ", err_T )
    println("Error in q:  ", err_q)
    println("Error in qx: ", err_qx)
    println("Error in qy: ", err_qy)

    # Visualise
    println("Visualisation:")
    @time PlotMakie( mesh, Te, xmin, xmax, ymin, ymax; cmap=:jet1, min_v=0.7, max_v = 1.5 )

    return mesh.nf, err_T, err_qx, err_qy, err_q
end

n         = 8
order     = 1
mesh_type = "Quadrangles"
main( n, mesh_type, order )