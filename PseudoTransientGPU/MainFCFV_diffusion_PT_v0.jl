const USE_GPU = false # ACHTUNG -> uncomment PT loop !!
const USE_DIRECT = false

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

function StabParam(τ, dA, Vol, mesh_type) # ACHTUNG does not work on GPU (called from woithin kernel !)
    if mesh_type=="Quadrangles";        τi = τ;    end
    # if mesh_type=="UnstructTriangles";  τi = τ*dA; end
    if mesh_type=="UnstructTriangles";  τi = τ end
    return τi
end

#--------------------------------------------------------------------#
    
@views function main()

    println("\n******** FCFV POISSON ********")
    # Create sides of mesh
    n          = 1
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = n*20, n*20
    R          = 0.5
    inclusion  = 0
    # mesh_type  = "Quadrangles"
    mesh_type  = "UnstructTriangles"
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        τr   = 1
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R )
    elseif mesh_type=="UnstructTriangles"  
        τr   = 100
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
    @time αe, βe, Ζe = ComputeFCFV(mesh, se, Tdir, τr)

    if USE_DIRECT
        # Assemble element matrices and RHS
        println("Compute element matrices:")
        @time Kv, fv = ElementAssemblyLoop(mesh, αe, βe, Ζe, Tdir, Tneu, τr)

        # Assemble triplets and sparse
        println("Assemble triplets and sparse:")
        @time K, f = CreateTripletsSparse(mesh, Kv, fv)
        # display(UnicodePlots.spy(K))

        # Solve for hybrid variable
        println("Direct solve:")
        # @time Th = K\f
        PC  = 0.5 .* (K .+ transpose(K))
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
        # Compute residual on faces -  This is check
        @time Te, qx, qy = ComputeElementValues(mesh, Th, αe, βe, Ζe, Tdir, τr)
        @time F          = ResidualOnFaces(mesh, Th, Te, qx, qy, τr)
        # println(F)
        println("Norm of matrix-free residual: ", norm(F)/length(F))
        Te2 = copy(Te)
    end

    if USE_GPU
        @show maxthreads = mesh.nf
        BLOC = 256
        GRID = Int(ceil(maxthreads/BLOC))
        @show cuthreads = BLOC
        @show cublocks  = GRID

        Th    = CUDA.zeros(Float64, mesh.nf)
        Te    = CUDA.zeros(Float64, mesh.nel)
        qx    = CUDA.zeros(Float64, mesh.nel)
        qy    = CUDA.zeros(Float64, mesh.nel)
        ΔTΔτ  = CUDA.zeros(Float64, mesh.nf)
        ΔTΔτ0 = CUDA.zeros(Float64, mesh.nf)
        Th_PT = CUDA.zeros(Float64, mesh.nf)
        ae    = CuArray(ae)
        be    = CuArray(be)
        ze    = CuArray(ze)
        Tneu  = CuArray(Tneu)

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
        ΔTΔτ  = zeros(mesh.nf)
        ΔTΔτ0 = zeros(mesh.nf)
        Th_PT = zeros(mesh.nf)
    end

    # Now a PT solve
    θ       = 1.0/(3.0*pi)
    Δτ      = 3.5*minimum(mesh.dA)/1.01

    # θ       = 1.0/(6*pi)
    # Δτ      = 3.5*minimum(mesh.dA)/1.01*2.9 # n=2

    # θ       = 1.0/(12.0*pi)
    # Δτ      = 3.5*minimum(mesh.dA)/1.01*2.9*2.9 # n=4

    nout    = 1e2
    iterMax = 5e4
    ϵ_PT    = 1e-7
    # println(minimum(mesh.dA))
    # println(maximum(mesh.dA))

    # PT loop
    @time for iter=1:iterMax
         if USE_GPU
              # @cuda blocks=cublocks threads=cuthreads ResidualOnFaces_v2_GPU!(F, Mesh_bc, Mesh_f2e, Mesh_dA_f, Mesh_n_x_f, Mesh_n_y_f, Mesh_vole_f, Mesh_vole, Mesh_e2f, Mesh_dA, Mesh_n_x, Mesh_n_y, Th_PT, Te, Tneu, qx, qy, ae, be, ze, τ, mesh_nf, mesh_nf_el) #mesh_type not ok because string
              # synchronize()
              # @cuda blocks=cublocks threads=cuthreads Update_F_GPU!(F, Th_PT, ΔTΔτ0, Δτ, θ, mesh_nf)
              # synchronize()
         else
            ΔTΔτ0 .= ΔTΔτ 
            ResidualOnFaces_v2!(ΔTΔτ, mesh, Th_PT, Te, qx, qy, αe, βe, Ζe, τr, Tneu)
            ΔTΔτ  .= (1.0 - θ).*ΔTΔτ0 .+ ΔTΔτ 
            Th_PT .= Th_PT .+ Δτ .* ΔTΔτ
         end
        if iter % nout == 0
            err = norm(ΔTΔτ)/sqrt(length(ΔTΔτ))
            println("PT Iter. ", iter, " --- Norm of matrix-free residual: ", err)
            if err < ϵ_PT
                print("PT solve converged in")
                break
            elseif isnan(err)
                error("NaN !")
            end
        end
    end

    Th .= Th_PT

    # Reconstruct element values
    println("Compute element values:")
    @time Te, qx, qy = ComputeElementValues(mesh, Array(Th), Array(αe), Array(βe), Array(Ζe), Array(Tdir), τr)
    @time Te1, qx1, qy1 = ComputeElementValuesFaces(mesh, Array(Th), Array(αe), Array(βe), Array(Ζe), Array(Tdir), τr)
    if USE_DIRECT
        println(norm(Te.-Te2)/length(Te[:]))
    end
    println(norm(Te.-Te1)/length(Te[:]))
    println(norm(qx.-qx1)/length(qx[:]))
    println(norm(qy.-qy1)/length(qy[:]))

    # Compute discretisation errors
    err_T, err_qx, err_qy = ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    println("Error in T:  ", err_T )
    println("Error in qx: ", err_qx)
    println("Error in qy: ", err_qy)

    # Visualise
    # print("Visualisation:")
    # if USE_DIRECT
    #     @time PlotMakie(mesh, Te,  xmin, xmax, ymin, ymax, :batlow, 0.9, 1.2)
    # else
    #     @time PlotMakie(mesh, Te1, xmin, xmax, ymin, ymax, :batlow, 0.9, 1.2)
    #     # @time PlotMakie(mesh, Te1, xmin, xmax, ymin, ymax; cmap = :batlow, min_v = 0.9, max_v = 1.2)
    # end
    print("Done! Total runtime:")
    return
end

@time main()
