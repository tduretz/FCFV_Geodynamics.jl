const USE_GPU = false # ACHTUNG -> uncomment PT loop !!
const USE_DIRECT = false

using Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT

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

function StabParam(τ, Γ, Ω, mesh_type, ν) # ACHTUNG does not work on GPU (called from within kernel !)
    if mesh_type=="Quadrangles";        τi = τ;    end
    # if mesh_type=="UnstructTriangles";  τi = τ*dA; end
    if mesh_type=="UnstructTriangles";  τi = τ end
    return τi
end

#--------------------------------------------------------------------#
    
@views function main( mesh_type, n, θ, Δτ )

    println("\n******** FCFV POISSON ********")
    # Create sides of mesh
    # n          = 2
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = Int16(n*20), Int16(n*20)
    R          = 0.5
    inclusion  = 0
    # mesh_type  = "Quadrangles"
    # mesh_type  = "UnstructTriangles"
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        τr   = 1
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R )
    elseif mesh_type=="UnstructTriangles"  
        τr   = 1
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R ) 
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
    @time αe, βe, Ζe = ComputeFCFV(mesh, se, Tdir)

    if USE_DIRECT
        # Assemble element matrices and RHS
        println("Compute element matrices:")
        @time Kv, fv = ElementAssemblyLoop(mesh, αe, βe, Ζe, Tdir, Tneu)

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
        @time Te, qx, qy = ComputeElementValues(mesh, Th, αe, βe, Ζe, Tdir)
        @time F          = ResidualOnFaces(mesh, Th, Te, qx, qy)
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

    # # Now a PT solve
    # θ       = 4.0*minimum(mesh.Γ)# 1.0/(3.0*pi)
    # # Δτ      = 3.5*minimum(mesh.Γ)/1.01


    # θ       = 2.9*minimum(mesh.Γ)# 1.0/(6*pi)
    # Δτ      = 3.5*minimum(mesh.Γ)/1.01*2.9 # n=2

    # println(θ)
    # println(Δτ)

    # θ       = 1.0/(12.0*pi)
    # Δτ      = 3.5*minimum(mesh.Γ)/1.01*2.9*2.9 # n=4

    nout    = 50#1e1
    iterMax = 5e4
    ϵ_PT    = 1e-7
    println(minimum(mesh.Γ))
    println(maximum(mesh.Γ))
    println(minimum(mesh.Ω))
    println(maximum(mesh.Ω))
    println("Δτ = ", Δτ)
    Ωe = maximum(mesh.Ω)
    Δx = minimum(mesh.Γ)
    D  = 1.0
    println("Δτ1 = ", Δx^2/(1.1*D) * 1.0/Ωe *2/3)

    # PT loop
    local iter = 0
    success = 0
    @time while (iter<iterMax)
        iter +=1
        if USE_GPU
            # @cuda blocks=cublocks threads=cuthreads ResidualOnFaces_v2_GPU!(F, Mesh_bc, Mesh_f2e, Mesh_dA_f, Mesh_n_x_f, Mesh_n_y_f, Mesh_vole_f, Mesh_vole, Mesh_e2f, Mesh_dA, Mesh_n_x, Mesh_n_y, Th_PT, Te, Tneu, qx, qy, ae, be, ze, τ, mesh_nf, mesh_nf_el) #mesh_type not ok because string
            # synchronize()
            # @cuda blocks=cublocks threads=cuthreads Update_F_GPU!(F, Th_PT, ΔTΔτ0, Δτ, θ, mesh_nf)
            # synchronize()
        else
        ΔTΔτ0 .= ΔTΔτ 
        ResidualOnFaces_v2!(ΔTΔτ, mesh, Th_PT, αe, βe, Ζe, Tneu)
        ΔTΔτ  .= (1.0 - θ).*ΔTΔτ0 .+ ΔTΔτ 
        Th_PT .= Th_PT .+ Δτ .* ΔTΔτ
        end
        if iter % nout == 0
            err = norm(ΔTΔτ)/sqrt(length(ΔTΔτ))
            println("PT Iter. ", iter, " --- Norm of matrix-free residual: ", err)
            if err < ϵ_PT
                print("PT solve converged in")
                success = true
                break
            elseif err>1e4
                success = false
                println("exploding !")
                break
            elseif isnan(err)
                success = false
                println("NaN !")
                break
            end
        end
    end

    Th .= Th_PT

    # Reconstruct element values
    println("Compute element values:")
    @time Te, qx, qy = ComputeElementValues(mesh, Array(Th), Array(αe), Array(βe), Array(Ζe), Array(Tdir))
    @time Te1, qx1, qy1 = ComputeElementValuesFaces(mesh, Array(Th), Array(αe), Array(βe), Array(Ζe))
    if USE_DIRECT
        println(norm(Te.-Te2)/length(Te[:]))
        println(norm(Te.-Te1)/length(Te[:]))
        println(norm(qx.-qx1)/length(qx[:]))
        println(norm(qy.-qy1)/length(qy[:]))
    end

    # Compute discretisation errors
    err_T, err_qx, err_qy = ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    println("Error in T:  ", err_T )
    println("Error in qx: ", err_qx)
    println("Error in qy: ", err_qy)

    # Visualise
    # print("Visualisation:")
    # if USE_DIRECT
        # @time PlotMakie(mesh, Te,  xmin, xmax, ymin, ymax, cgrad(:roma, rev=true), 0.7, 1.5)
    # else
    #     @time PlotMakie(mesh, Te1, xmin, xmax, ymin, ymax, :batlow, 0.9, 1.2)
        # @time PlotMakie(mesh, Te1, xmin, xmax, ymin, ymax; cmap = :batlow, min_v = 0.9, max_v = 1.2)
    # end
    print("Done! Total runtime:")
    return iter, success, length(Th), nx
end

#########################################################

# quads
# main( "Quadrangles", 1, 0.19, 0.88 )
# main( "Quadrangles",2, 0.11, 0.93 )
# main( "Quadrangles",4, 0.057, 0.965 )
# main( "Quadrangles",8, 0.029, 0.125*7.25 )
# main( "Quadrangles",16, 0.014125, 0.125*7.25*1.0 ) #--> 1900

# triangles (test were made with τ=100 which is inappropriate, need to remake)
# main( "UnstructTriangles", 1.25, 0.095, 0.145 )
# main( "UnstructTriangles",3, 0.04, 0.21 )
# main( "UnstructTriangles",1.5, 0.082, 0.1575 )
# main( "UnstructTriangles",6, 0.02, 0.24 )
# main( "UnstructTriangles",8, 0.014, 0.25 )
# main( "UnstructTriangles",2, 0.060, 0.180 )
# main( "UnstructTriangles",4, 0.03, 0.22 )
# main( "UnstructTriangles",10, 0.011, 0.256 )
# main( "UnstructTriangles", 1, 0.12, 0.125 )


# nv  = [1, 2, 4, 8]
# tet  = 0.01:0.01/2:0.1
# dtau =  0.1:0.05:1

# sucv = zeros(length(nv), length(tet), length(dtau))
# itv  = zeros(length(nv), length(tet), length(dtau))

# for in=1:length(nv)
#     for iθ=1:length(tet)
#         for iΔ=1:length(dtau)
#             @printf("n = %02d -- θ = %2.3f -- Δτ = %2.3f\n", nv[in], tet[iθ], dtau[iΔ] )
#             iter, success = main( nv[in], tet[iθ], dtau[iΔ] )
#             if success==true sucv[in,iθ,iΔ] = 1 end
#             itv[in,iθ,iΔ]  = iter
#         end
#     end
# end

# file = matopen(string(@__DIR__,"/PT_syst.mat"), "w")
# write(file, "n", nv)
# write(file, "tet", Array(tet))
# write(file, "dtau", Array(dtau))
# write(file, "success", sucv)
# write(file, "iter", itv)
# close(file)

main( "UnstructTriangles", 1, 0.11428, 0.28 )
main( "UnstructTriangles", 2, 0.11428/2, 0.28 )
main( "UnstructTriangles", 4, 0.11428/4, 0.28 )
main( "UnstructTriangles", 8, 0.11428/8.0, 0.28 )
main( "UnstructTriangles", 16, 0.11428/16, 0.28 )
