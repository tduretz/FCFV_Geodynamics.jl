const USE_MAKIE    = false
# import Plots
# include("../VisuFCFV.jl")

using Printf, LinearAlgebra
using CUDA
using CellArrays, StaticArrays

include("helpers.jl")
include("../CreateMeshFCFV.jl")
include("../EvalAnalDani.jl")

@inbounds function glob2loc!(V,P,Vx,Vy,Pr,e2n,nel,nrange)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        nodes = e2n[ie]
        for in ∈ nrange
            iloc1, iloc2 = in*2-1, in*2
            field(V,iloc1)[ie] = Vx[nodes[in]]
            field(V,iloc2)[ie] = Vy[nodes[in]]
        end
        field(P,1)[ie]  = Pr[ie]
    end
    return nothing
end

@inbounds function elem_op!(Fv,Fp,K,Q,V,P,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        Fv[ie] = K[ie]*V[ie] .+ Q[ie]  * P[ie]
        Fp[ie] =             .- Q[ie]' * V[ie]
    end
    return nothing
end

@inbounds function loc2glob!(Fx,Fy,FP,Fv,Fp,e2n,nel,nrange)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        nodes = e2n[ie]
        for in ∈ nrange
            iloc1, iloc2 = in*2-1, in*2
            CUDA.@atomic Fx[nodes[in]] -= Fv[ie][iloc1]
            CUDA.@atomic Fy[nodes[in]] -= Fv[ie][iloc2]
        end
        FP[ie] -= Fp[ie][1]
    end
    return nothing
end

@inbounds function init_CellArrays!(K,Q,E2N,K_glob,Q_glob,e2n,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        for c2 = 1:cellsize(K,2), c1 = 1:cellsize(K,1)
            field(K,c1,c2)[ie] = K_glob[ie,c1,c2]
            field(Q,c1   )[ie] = Q_glob[ie,c1]
        end
        for c1 = 1:cellsize(E2N,1)
            field(E2N,c1)[ie] = e2n[ie,c1]
        end
    end
    return nothing
end

@inbounds function update_nodes_v!(Vx,Vy,ΔVxΔτ0,ΔVyΔτ0,ΔVxΔτ,ΔVyΔτ,θ,ΔτV,nn)
    in = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if in <= nn
        ΔVxΔτ[in]  = (1.0 - θ).*ΔVxΔτ0[in] + ΔVxΔτ[in]
        ΔVyΔτ[in]  = (1.0 - θ).*ΔVyΔτ0[in] + ΔVyΔτ[in]
        Vx[in]     = Vx[in] + ΔτV * ΔVxΔτ[in]
        Vy[in]     = Vy[in] + ΔτV * ΔVyΔτ[in]
        ΔVxΔτ0[in] = ΔVxΔτ[in]
        ΔVyΔτ0[in] = ΔVyΔτ[in]
        ΔVxΔτ[in]  = 0.0
        ΔVyΔτ[in]  = 0.0
    end
    return nothing
end

@inbounds function update_nodes_p!(Pr,ΔPΔτ,ΔτP,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        Pr[ie]   = Pr[ie] + ΔτP * ΔPΔτ[ie]
        ΔPΔτ[ie] = 0.0
    end
    return nothing
end


@inbounds function set_bc_nodes_v!(ΔVxΔτ,ΔVyΔτ,bcn,nn)
    in = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if in <= nn
        if (bcn[in]==1) ΔVxΔτ[in] = 0.0  end
        if (bcn[in]==1) ΔVyΔτ[in] = 0.0  end
    end
    return nothing
end

@inbounds function postprocess!(Vxe,Vye,Pe,Vx,Vy,Pr,e2n,nnel,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        for in = 1:nnel
            CUDA.@atomic Vxe[ie] += 1.0/nnel * Vx[e2n[ie,in]]
            CUDA.@atomic Vye[ie] += 1.0/nnel * Vy[e2n[ie,in]]
        end
        Pe[ie] = Pr[ie] # if npel==3 Pe[ie] = 1.0/npel * (Array(Pr)[ie] + Array(Pr)[e+mesh.nel] + Array(Pr)[ie+2*mesh.nel]) end
    end
    return nothing
end

function main(n, nnel, npel, nip, θ, ΔτV, ΔτP)
    println("\n******** FEM Stokes ********")
    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    nx, ny     = Int16(n*16), Int16(n*16)
    R          = 1.0
    inclusion  = 1
    η          = (1.0, 5.0)
    # Generate mesh
    mesh = MakeTriangleMesh(nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R; nnel, npel)
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    println("Number of MDofs: ", (mesh.nn*2+mesh.nel)/1e6)
    nel, nn, npel, nnel = mesh.nel, mesh.nn, mesh.npel, mesh.nnel
    # Numerics
    nout       = 2000
    iterMax    = 2e5
    ϵ_PT       = 1e-7
    threads    = 256
    blocks_e   = nel÷threads + 1
    blocks_n   =  nn÷threads + 1
    nrange     = 1:nnel
    # Initial condition
    mesh.ke[mesh.phase.==1] .= η[1]
    mesh.ke[mesh.phase.==2] .= η[2]
    Vx         = zeros(mesh.nn)       # Solution on nodes
    Vy         = zeros(mesh.nn)       # Solution on nodes
    Pr         = zeros(mesh.nel*npel) # Solution on elements
    Pa         = zeros(mesh.nel)      # Solution on elements
    # Intial guess from analytics
    for in = 1:mesh.nn
        if (mesh.bcn[in]==1)  Vx[in],Vy[in],_ = EvalAnalDani(mesh.xn[in], mesh.yn[in], R, η[1], η[2]) end
    end
    for ie = 1:mesh.nel
        _,_,Pa[ie] = EvalAnalDani(mesh.xc[ie], mesh.yc[ie], R, η[1], η[2])
    end
    # Element data
    ipx, ipw   = IntegrationTriangle(nip)
    N, dNdX    = ShapeFunctions(ipx, nip, nnel)
    K_all, Q_all = ElementAssemblyLoopFEM(mesh, ipw, N, dNdX)
    # Array init
    ΔVxΔτ      = CUDA.zeros(nn)
    ΔVyΔτ      = CUDA.zeros(nn)
    ΔVxΔτ0     = CUDA.zeros(nn)
    ΔVyΔτ0     = CUDA.zeros(nn)
    ΔPΔτ       = CUDA.zeros(nel*npel)
    e2n        = CuArray(mesh.e2n) #(nel, nnel)
    bcn        = CuArray(mesh.bcn) #nn
    Vx         = CuArray(Vx)       #nn
    Vy         = CuArray(Vy)       #nn
    Pr         = CuArray(Pr)       #nel*npel
    K_all      = CuArray(K_all)    #(nel, nnel*2, nnel*2)
    Q_all      = CuArray(Q_all)    #(nel, nnel*2, 1)    
    # Local cell array size
    dims_nel   = (nel)    
    celldims_K_l = (nnel*2, nnel*2)
    celldims_Q_l = (nnel*2, 1)
    celldims_V_l = (nnel*2, 1)
    celldims_P_l = (npel  , 1)
    celldims_e2n_l = (nnel  , 1)
    # Init FEM tmp CellArrays
    K_loc  = SMatrix{celldims_K_l..., Float64, prod(celldims_K_l)}
    K      = CuCellArray{K_loc}(undef, dims_nel)
    Q_loc  = SMatrix{celldims_Q_l..., Float64, prod(celldims_Q_l)}
    Q      = CuCellArray{Q_loc}(undef, dims_nel)
    V_loc  = SMatrix{celldims_V_l..., Float64, prod(celldims_V_l)}
    V      = CuCellArray{V_loc}(undef, dims_nel)
    P_loc  = SMatrix{celldims_P_l..., Float64, prod(celldims_P_l)}
    P      = CuCellArray{P_loc}(undef, dims_nel)
    Fv_loc = SMatrix{celldims_V_l..., Float64, prod(celldims_V_l)}
    Fv     = CuCellArray{Fv_loc}(undef, dims_nel)
    Fp_loc = SMatrix{celldims_P_l..., Float64, prod(celldims_P_l)}
    Fp     = CuCellArray{Fp_loc}(undef, dims_nel)
    e2n_l  = SMatrix{celldims_e2n_l..., Int64, prod(celldims_e2n_l)}
    E2N    = CuCellArray{e2n_l}(undef, dims_nel)
    # PT Loop
    local iter = 0; success = 0
    @cuda blocks=blocks_e threads=threads init_CellArrays!(K,Q,E2N,K_all,Q_all,e2n,nel); synchronize()
    @time while (iter<iterMax)
        iter += 1
        @cuda blocks=blocks_e threads=threads glob2loc!(V,P,Vx,Vy,Pr,E2N,nel,nrange); synchronize()
        @cuda blocks=blocks_e threads=threads elem_op!(Fv,Fp,K,Q,V,P,nel); synchronize()
        @cuda blocks=blocks_e threads=threads loc2glob!(ΔVxΔτ,ΔVyΔτ,ΔPΔτ,Fv,Fp,E2N,nel,nrange); synchronize()
        @cuda blocks=blocks_n threads=threads set_bc_nodes_v!(ΔVxΔτ,ΔVyΔτ,bcn,nn); synchronize()
        # Check error
        if iter % nout == 0 || iter==1
            errVx = norm(ΔVxΔτ)/sqrt(length(ΔVxΔτ))
            errVy = norm(ΔVyΔτ)/sqrt(length(ΔVyΔτ))
            errP  = norm(ΔPΔτ)/sqrt(length(ΔPΔτ))
            @printf("PT Iter. %05d:\n", iter)
            @printf("  ||Fx|| = %3.3e\n", errVx)
            @printf("  ||Fy|| = %3.3e\n", errVy)
            @printf("  ||Fp|| = %3.3e\n", errP )
            err = max(errVx, errVy, errP)
            if err < ϵ_PT     print("PT solve converged in"); success = true;  break
            elseif err>1e4    println("exploding !");         success = false; break
            elseif isnan(err) println("NaN !");               success = false; break
            end
        end
        @cuda blocks=blocks_n threads=threads update_nodes_v!(Vx,Vy,ΔVxΔτ0,ΔVyΔτ0,ΔVxΔτ,ΔVyΔτ,θ,ΔτV,nn); synchronize()
        @cuda blocks=blocks_e threads=threads update_nodes_p!(Pr,ΔPΔτ,ΔτP,nel); synchronize()
    end
    # Postprocessing
    Vxe = CUDA.zeros(nel)
    Vye = CUDA.zeros(nel)
    Pe  = CUDA.zeros(nel)
    @cuda blocks=blocks_e threads=threads postprocess!(Vxe,Vye,Pe,Vx,Vy,Pr,e2n,nnel,nel); synchronize()

    @printf("min, max Vx: %2.2e %2.2e\n", minimum(Vx), maximum(Vx))
    @printf("min, max Vy: %2.2e %2.2e\n", minimum(Vy), maximum(Vy))
    @printf("min, max Pr: %2.2e %2.2e\n", minimum(Pr), maximum(Pr))
    #-----------------------------------------------------------------#
    # if USE_MAKIE
    #     # PlotMakie(mesh, Vxe, xmin, xmax, ymin, ymax; cmap=:jet1)
    #     PlotMakie(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1 )
    # end
    return nothing
end

# main(2, 7, 1, 6, 0.030598470000000003, 0.03666666667,  1.0) # nit = 4000
# main(4, 7, 1, 6, 0.03/2, 0.036, 1.5) # nit = 9000
# main(8, 7, 1, 6, 0.03/3, 0.036, 1.8) # nit = 19000

main(18, 7, 1, 6, 0.03/5, 0.036, 6.0) # nit = 14000 <- MILAMIN/4

# main(100, 7, 1, 6, 0.03/5, 0.036, 6.0) # nit =  <- not yet BILAMIN
