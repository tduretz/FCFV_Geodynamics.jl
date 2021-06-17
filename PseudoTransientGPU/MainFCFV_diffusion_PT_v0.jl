const USE_GPU = true

import TriangleMesh
using Printf, LoopVectorization, LinearAlgebra, SparseArrays

include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../DiscretisationFCFV.jl")
if USE_GPU
    using CUDA
    include("ResidualEvalFCFV_PT_GPU.jl")
else
    include("ResidualEvalFCFV_PT.jl")
end

#--------------------------------------------------------------------#

function SetUpProblem!( mesh, T, Tdir, Tneu, se, a, b, c, d, alp, bet )
    # Evaluate T analytic on cell faces
    @avx for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        Tdir[in] = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        dTdx     = Tdir[in] * (a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y))
        dTdy     = Tdir[in] * (alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y))
        Tneu[in] = -dTdy # nx*dTdx + nt*dTdy on SOUTH face
    end
    # Evaluate T analytic on barycentres
    @avx for iel=1:mesh.nel
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

#--------------------------------------------------------------------#    

function StabParam(tau, dA, Vol, mesh_type)
    if mesh_type=="Quadrangles";        taui = tau;    end
    # if mesh_type=="UnstructTriangles";  taui = tau*dA; end
    if mesh_type=="UnstructTriangles";  taui = tau end
    return taui
end

#--------------------------------------------------------------------#
    
@views function main()

    println("\n******** FCFV POISSON ********")

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = 200, 200
    R          = 0.5
    inclusion  = 0
    # mesh_type  = "Quadrangles"
    mesh_type  = "UnstructTriangles"
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        tau  = 1
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R )
    elseif mesh_type=="UnstructTriangles"  
        tau  = 100
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R ) 
    end
    println("Number of elements: ", mesh.nel)

    # Source term and BCs etc...
    Tanal  = zeros(mesh.nel)
    se     = zeros(mesh.nel)
    Tdir   = zeros(mesh.nf)
    Tneu   = zeros(mesh.nf)
    alp = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;
    println("Model configuration :")
    @time SetUpProblem!(mesh , Tanal, Tdir, Tneu, se, a, b, c, d, alp, bet)

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    @time ae, be, ze = ComputeFCFV(mesh, se, Tdir, tau)

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

    # # Compute residual on faces -  This is check
    # @time Te, qx, qy = ComputeElementValues(mesh, Th, ae, be, ze, Tdir, tau)
    # @time F = ResidualOnFaces(mesh, Th, Te, qx, qy, tau)
    # # println(F)
    # println("Norm of matrix-free residual: ", norm(F)/length(F))

    if USE_GPU
        @show maxthreads = mesh.nf
        BLOC = 256
        GRID = Int(ceil(maxthreads/BLOC))
        @show cuthreads = BLOC
        @show cublocks  = GRID

        Th    = CUDA.zeros(mesh.nf)
        Te    = CUDA.zeros(mesh.nel)
        qx    = CUDA.zeros(mesh.nel)
        qy    = CUDA.zeros(mesh.nel)
        F     = CUDA.zeros(mesh.nf)
        F0    = CUDA.zeros(mesh.nf)
        Th_PT = CUDA.zeros(mesh.nf)
        ae    = CuArray(ae)
        be    = CuArray(be)
        ze    = CuArray(ze)

        mesh_nf     = mesh.nf
        mesh_type   = mesh.type
        mesh_nf_el  = mesh.nf_el
        Mesh_bc     = CuArray(mesh.bc)
        Mesh_f2e    = CuArray(mesh.f2e)
        Mesh_dA_f   = CuArray(mesh.dA_f)
        Mesh_n_x_f  = CuArray(mesh.n_x_f)
        Mesh_n_y_f  = CuArray(mesh.n_y_f)
        Mesh_vole_f = CuArray(mesh.vole_f)
        Mesh_vole   = CuArray(mesh.vole)
        Mesh_e2f    = CuArray(mesh.e2f)
        Mesh_dA     = CuArray(mesh.dA)
        Mesh_n_x    = CuArray(mesh.n_x)
        Mesh_n_y    = CuArray(mesh.n_y)
    else
        Th    = zeros(mesh.nf)
        Te    = zeros(mesh.nel)
        qx    = zeros(mesh.nel)
        qy    = zeros(mesh.nel)
        F     = zeros(mesh.nf)
        F0    = zeros(mesh.nf)
        Th_PT = zeros(mesh.nf)
    end

    # Now a PT solve
    dmp     = 0.7 # 0.24
    dTdtau  = 0.02
    nout    = 1e3
    iterMax = 1e6

    # PT loop
    @time for iter=1:iterMax
        if USE_GPU
             @cuda blocks=cublocks threads=cuthreads ResidualOnFaces_v2_GPU!(F, Mesh_bc, Mesh_f2e, Mesh_dA_f, Mesh_n_x_f, Mesh_n_y_f, Mesh_vole_f, Mesh_vole, Mesh_e2f, Mesh_dA, Mesh_n_x, Mesh_n_y, Th_PT, Te, qx, qy, ae, be, ze, tau, mesh_nf, mesh_nf_el) #mesh_type not ok because string
             synchronize()
             @cuda blocks=cublocks threads=cuthreads Update_F_GPU!(F, Th_PT, F0, dTdtau, dmp, mesh_nf)
             synchronize()
        else
            ResidualOnFaces_v2!(F, mesh, Th_PT, Te, qx, qy, ae, be, ze, tau)
            F      .= (1 - dmp).*F0 .+ F                                       # to be updated with @avx
            Th_PT  .+= dTdtau.*F                                               # to be updated with @avx
            F0     .= F                                                        # to be updated with @avx
        end

        if iter % nout == 0
            println("PT Iter. ", iter, " --- Norm of matrix-free residual: ", norm(F)/length(F))
            if norm(F)/length(F) < 1e-6
                print("PT solve converged in")
                break
            end
        end
    end

    Th .= Th_PT

    # Reconstruct element values
    println("Compute element values:")
    @time Te, qx, qy = ComputeElementValues(mesh, Array(Th), Array(ae), Array(be), Array(ze), Array(Tdir), tau)
    @time Te1, qx1, qy1 = ComputeElementValuesFaces(mesh, Array(Th), Array(ae), Array(be), Array(ze), Array(Tdir), tau)
    println(norm(Te.-Te1)/length(Te[:]))
    println(norm(qx.-qx1)/length(qx[:]))
    println(norm(qy.-qy1)/length(qy[:]))

    # Compute discretisation errors
    err_T, err_qx, err_qy = ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    println("Error in T:  ", err_T )
    println("Error in qx: ", err_qx)
    println("Error in qy: ", err_qy)

    # Visualise
    println("Visualisation:")
    @time PlotMakie( mesh, Te1 )
    # PlotElements( mesh )

end

main()
