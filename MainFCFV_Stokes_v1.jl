include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV.jl")
using LoopVectorization
using SparseArrays, LinearAlgebra
import UnicodePlots 

function SetUpProblem!(mesh, P, Vx, Vy, VxDir, VxNeu, VyDir, VyNeu, sx, sy)
    # Evaluate T analytic on cell faces
    @avx for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        VxDir[in] = x^2*(1 - x)^2*(4*y^3 - 6*y^2 + 2*y)
        VyDir[in] =-y^2*(1 - y)^2*(4*x^3 - 6*x^2 + 2*x)
    end
    # Evaluate T analytic on barycentres
    @avx for iel=1:mesh.nel
        x       = mesh.xc[iel]
        y       = mesh.yc[iel]
        p       =  x*(1-x)
        P[iel]  =  p
        Vx[iel] =  x^2*(1 - x)^2*(4*y^3 - 6*y^2 + 2*y)
        Vy[iel] = -y^2*(1 - y)^2*(4*x^3 - 6*x^2 + 2*x)
        sx[iel] = -p^2*(24*y - 12) - 4*x^2*(4*y^3 - 6*y^2 + 2*y) - 8*x*(2*x - 2)*(4*y^3 - 6*y^2 + 2*y) - 2*x + 1.0*y^2*(2*y - 2)*(12*x^2 - 12*x + 2) + 2.0*y*(1 - y)^2*(12*x^2 - 12*x + 2) - 4*(1 - x)^2*(4*y^3 - 6*y^2 + 2*y) + 1
        sy[iel] = -2*p*(1 - x)*(12*y^2 - 12*y + 2) - x^2*(2*x - 2)*(12*y^2 - 12*y + 2) + 1.0*y^2*(1 - y)^2*(24*x - 12) + 4*y^2*(4*x^3 - 6*x^2 + 2*x) + 8*y*(2*y - 2)*(4*x^3 - 6*x^2 + 2*x) + 4*(1 - y)^2*(4*x^3 - 6*x^2 + 2*x)

    end
    return
end

function ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    eT  = zeros(mesh.nel)
    eqx = zeros(mesh.nel)
    eqy = zeros(mesh.nel)
    Ta  = zeros(mesh.nel)
    qxa = zeros(mesh.nel)
    qya = zeros(mesh.nel)
    @avx for iel=1:mesh.nel
        x        = mesh.xc[iel]
        y        = mesh.yc[iel]
        Ta[iel]       = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        qxa[iel]      = -Ta[iel] * (a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y))
        qya[iel]      = -Ta[iel] * (alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y))
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
    # if mesh_type=="UnstructTriangles";  taui = tau*dA; end
    if mesh_type=="UnstructTriangles";  taui = tau end
    return taui
end
    
@views function main()

    println("\n******** FCFV STOKES ********")

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = 8, 8
    mesh_type  = "Quadrangles"
    mesh_type  = "UnstructTriangles"
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        tau  = 1
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax )
    elseif mesh_type=="UnstructTriangles"  
        tau  = 100
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax ) 
    end
    println("Number of elements: ", mesh.nel)

    # Source term and BCs etc...
    Pa     = zeros(mesh.nel)
    Vxa    = zeros(mesh.nel)
    Vya    = zeros(mesh.nel)
    sex    = zeros(mesh.nel)
    sey    = zeros(mesh.nel)
    VxDir  = zeros(mesh.nf)
    VxNeu  = zeros(mesh.nf)
    VyDir  = zeros(mesh.nf)
    VyNeu  = zeros(mesh.nf)
    println("Model configuration :")
    @time SetUpProblem!(mesh, Pa, Vxa, Vya, VxDir, VxNeu, VyDir, VyNeu, sex, sey)

    # # Compute some mesh vectors 
    # println("Compute FCFV vectors:")
    # @time ae, be, ze = ComputeFCFV(mesh, se, Tdir, tau)

    # # Assemble element matrices and RHS
    # println("Compute element matrices:")
    # @time Kv, fv = ElementAssemblyLoop(mesh, ae, be, ze, Tdir, Tneu, tau)

    # # Assemble triplets and sparse
    # println("Assemble triplets and sparse:")
    # @time K, f = CreateTripletsSparse(mesh, Kv, fv)
    # # display(UnicodePlots.spy(K))

    # # Solve for hybrid variable
    # println("Direct solve:")
    # # @time Th   = K\f
    # PC  = 0.5.*(K.+K')
    # PCc = cholesky(PC)
    # Th  = zeros(mesh.nf)
    # dTh = zeros(mesh.nf,1)
    # r   = zeros(mesh.nf,1)
    # r  .= f - K*Th
    # # @time Th   = K\f
    # for rit=1:5
    #     r    .= f - K*Th
    #     println("It. ", rit, " - Norm of residual: ", norm(r)/length(r))
    #     if norm(r)/length(r) < 1e-10
    #         break
    #     end
    #     dTh  .= PCc\r
    #     Th  .+= dTh[:]
    # end

    # # Reconstruct element values
    # println("Compute element values:")
    # @time Te, qx, qy = ComputeElementValues(mesh, Th, ae, be, ze, Tdir, tau)

    # # Compute discretisation errors
    # err_T, err_qx, err_qy = ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    # println("Error in T:  ", err_T )
    # println("Error in qx: ", err_qx)
    # println("Error in qy: ", err_qy)

    # Visualise
    println("Visualisation:")
    @time PlotMakie( mesh, Pa )
    # PlotElements( mesh )

end

main()