include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV.jl")
using LoopVectorization
using SparseArrays, LinearAlgebra
import UnicodePlots 
using AbstractPlotting # this is not really good since it induces a warning

function SetUpProblem!( mesh, T, Tdir, Tneu, se, k, R )
    # precompute inverses
    ik1, ik2 = 1.0/k[1], 1.0/k[2]
    k1,  k2  = k[1], k[2]
    # Evaluate T analytic on cell faces
    for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        T1       =  ik1 *(x^2 + y^2)^(5.0/2.0)
        T2       =  ik2 *(x^2 + y^2)^(5.0/2.0) + (ik1 - ik2)*R^5.0
        out      = (x^2 + y^2)>R^2
        Tdir[in] = (out==1) * T1 + (out!=1)  * T2
        dTdx1    = 5.0*ik1*x^1.0*(x^2.0 + y^2.0)^1.5
        dTdy1    = 5.0*ik1*y^1.0*(x^2.0 + y^2.0)^1.5
        Tneu[in] = -dTdy1      # nx*dTdx + nt*dTdy on face
    end
    mesh.ke = zeros(mesh.nel)
    phase   = zeros(mesh.nel)
    se_bis  = zeros(mesh.nel)
    # Evaluate T analytic on barycentres
    for iel=1:mesh.nel
        x            = mesh.xc[iel]
        y            = mesh.yc[iel]
        T1           = ik1 *(x^2.0 + y^2.0)^(5.0/2.0)
        T2           = ik2 *(x^2.0 + y^2.0)^(5.0/2.0) + ( ik1 - ik2 )*R^5.0
        out          = (x^2 + y^2)>R^2
        T[iel]       = (out==1) * T1 + (out!=1)  * T2
        se[iel]      = -25.0*(x^2 + y^2)^(3.0/2.0)
        mesh.ke[iel] = (out==1) * k1 + (out!=1) * k2
        phase[iel]   = out
    end
    return
end

function ComputeError( mesh, Te, qx, qy, k, R )
    # precompute inverses
    ik1, ik2 = 1.0/k[1], 1.0/k[2]
    k1,  k2  = k[1], k[2]
    eT  = zeros(mesh.nel)
    eqx = zeros(mesh.nel)
    eqy = zeros(mesh.nel)
    Ta  = zeros(mesh.nel)
    qxa = zeros(mesh.nel)
    qya = zeros(mesh.nel)
     for iel=1:mesh.nel
        x        = mesh.xc[iel]
        y        = mesh.yc[iel]
        T1       = ik1 *(x^2.0 + y^2.0)^(5.0/2.0)
        T2       = ik2 *(x^2.0 + y^2.0)^(5.0/2.0) + ( ik1 - ik2 )*R^5.0
        out      = (x^2 + y^2)>R^2
        dTdx1    = 5.0*ik1*x^1.0*(x^2.0 + y^2.0)^1.5
        dTdx2    = 5.0*ik2*x^1.0*(x^2.0 + y^2.0)^1.5
        dTdy1    = 5.0*ik1*y^1.0*(x^2.0 + y^2.0)^1.5
        dTdy2    = 5.0*ik2*y^1.0*(x^2.0 + y^2.0)^1.5
        Ta[iel]  = (out==1) * T1 + (out!=1)  * T2
        qxa[iel] = -mesh.ke[iel] * ( (out==1) * dTdx1 + (out!=1)  * dTdx2 )
        qya[iel] = -mesh.ke[iel] * ( (out==1) * dTdy1 + (out!=1)  * dTdy2 )
        eT[iel]  = Te[iel] - Ta[iel]
        eqx[iel] = qx[iel] - qxa[iel]
        eqy[iel] = qy[iel] - qya[iel]
    end
    errT  = norm(eT)/norm(Ta)
    errqx = norm(eqx)/norm(qxa)
    errqy = norm(eqy)/norm(qya)
    return errT, errqx, errqy
end
    

function StabParam(tau, dA, Vol, mesh_type)
    if mesh_type=="Quadrangles";        taui = tau;    end
    if mesh_type=="UnstructTriangles";  taui = tau end
    return taui
end
    
@views function main()

    println("\n******** FCFV POISSON ********")

    # Create sides of mesh
    xmin, xmax = -1.0, 1.0
    ymin, ymax = -1.0, 1.0
    n          = 4
    nx, ny     = n*8, n*8
    k          = [1.0 100]
    R          = 0.5
    # mesh_type  = "Quadrangles"
    mesh_type  = "UnstructTriangles"
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        tau  = 1
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax )
    elseif mesh_type=="UnstructTriangles"  
        tau  = 1
        mesh = MakeTriangleMeshInclusion( nx, ny, xmin, xmax, ymin, ymax, R ) 
    end
    println("Number of elements: ", mesh.nel)

    # Source term and BCs etc...
    Tanal  = zeros(mesh.nel)
    se     = zeros(mesh.nel)
    Tdir   = zeros(mesh.nf)
    Tneu   = zeros(mesh.nf)
    alp = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;
    println("Model configuration :")
    @time SetUpProblem!(mesh , Tanal, Tdir, Tneu, se, k, R)

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    @time ae, be, ze = ComputeFCFV(mesh, se, Tdir, tau)

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    @time Kv, fv = ElementAssemblyLoop(mesh, ae, be, ze, Tdir, Tneu, tau)

    # Assemble triplets and sparse
    println("Assemble triplets and sparse:")
    @time K, f = CreateTripletsSparse(mesh, Kv, fv)
    # display(UnicodePlots.spy(K))

    # Solve for hybrid variable
    println("Direct solve:")
    # @time Th   = K\f
    PC  = 0.5*(K.+K')
    PCc = cholesky(PC)
    Th  = zeros(mesh.nf)
    dTh = zeros(mesh.nf,1)
    r   = zeros(mesh.nf,1)
    r  .= f - K*Th
    for rit=1:5
        r    .= f - K*Th
        println("It. ", rit, " - Norm of residual: ", norm(r)/length(r))
        if norm(r)/length(r) < 1e-10
            break
        end
        dTh  .= PCc\r
        Th  .+= dTh[:]
    end

    # Reconstruct element values
    println("Compute element values:")
    @time Te, qx, qy = ComputeElementValues(mesh, Th, ae, be, ze, Tdir, tau)
    # @time Te, qx, qy = ComputeElementValuesFaces(mesh, Th, ae, be, ze, Tdir, tau)

    # Compute discretisation errors
    err_T, err_qx, err_qy = ComputeError( mesh, Te, qx, qy, k, R )
    println("Error in T:  ", err_T )
    println("Error in qx: ", err_qx)
    println("Error in qy: ", err_qy)

    # Visualise
    println("Visualisation:")
    @time PlotMakie( mesh,  mesh.phase )

end

main()