const USE_GPU      = true  # Not supported yet 
const USE_DIRECT   = false  # Sparse matrix assembly + direct solver
const USE_NODAL    = false  # Nodal evaluation of residual
const USE_PARALLEL = false  # Parallel residual evaluation
const USE_MAKIE    = false  # Visualisation 
# import Plots
# include("../VisuFCFV.jl")

using Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid
using MAT

using CUDA
using CellArrays, StaticArrays

include("../CreateMeshFCFV.jl")
include("../DiscretisationFCFV.jl")
include("../EvalAnalDani.jl")
include("IntegrationPoints.jl")

StabParam(τ, Γ, Ω, mesh_type, ν) = 0.0 # Stabilisation is only needed for FCFV

function ResidualStokesElementalSerialFEM!(Fx, Fy, Fp, V_ele, Vx, Vy, Pr, K_all, Q_all, e2n, bcn, nnel, npel, nel)
    @inbounds for ie = 1:nel
        nodes = e2n[ie,:]
        if npel==1 nodesP = [ie] end
        if npel==3 nodesP = [ie; ie+nel; ie+2*nel]  end
        V_ele[1:2:end-1] .= Vx[nodes]
        V_ele[2:2:end]   .= Vy[nodes]
        P_ele      = Pr[nodesP] 
        K_ele      = K_all[ie,:,:]
        Q_ele      = Q_all[ie,:,:]
        fv_ele     = K_ele*V_ele .+ Q_ele * P_ele # .- bu[e,:]
        fp_ele     =              .-Q_ele'* V_ele # .- bp[e,:]
        Fx[nodes] .-= fv_ele[1:2:end-1] # This should be atomic
        Fy[nodes] .-= fv_ele[2:2:end]
        if npel==1 Fp[nodesP]  -= fp_ele[:]    end
        if npel==3 Fp[nodesP] .-= fp_ele[:] end
    end
    Fx[bcn.==1] .= 0.0
    Fy[bcn.==1] .= 0.0
    return nothing
end

@inbounds function my_gpu_fun_0!(V,P,Vx,Vy,Pr,e2n,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        nodes = e2n[ie]
        field(V,1 )[ie] = Vx[nodes[1]]
        field(V,3 )[ie] = Vx[nodes[2]]
        field(V,5 )[ie] = Vx[nodes[3]]
        field(V,7 )[ie] = Vx[nodes[4]]
        field(V,9 )[ie] = Vx[nodes[5]]
        field(V,11)[ie] = Vx[nodes[6]]
        field(V,13)[ie] = Vx[nodes[7]]
        field(V,2 )[ie] = Vy[nodes[1]]
        field(V,4 )[ie] = Vy[nodes[2]]
        field(V,6 )[ie] = Vy[nodes[3]]
        field(V,8 )[ie] = Vy[nodes[4]]
        field(V,10)[ie] = Vy[nodes[5]]
        field(V,12)[ie] = Vy[nodes[6]]
        field(V,14)[ie] = Vy[nodes[7]]
        field(P,1)[ie]  = Pr[ie]
    end
    return nothing
end

@inbounds function my_gpu_fun_1!(Fv,Fp,K,Q,V,P,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        Fv[ie] = K[ie]*V[ie] .+ Q[ie]  * P[ie]
        Fp[ie] =             .- Q[ie]' * V[ie]
    end
    return nothing
end

@inbounds function my_gpu_fun_2!(Fx,Fy,FP,Fv,Fp,e2n,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        nodes = e2n[ie]
        CUDA.@atomic Fx[nodes[1]] -= Fv[ie][1]
        CUDA.@atomic Fx[nodes[2]] -= Fv[ie][3]
        CUDA.@atomic Fx[nodes[3]] -= Fv[ie][5]
        CUDA.@atomic Fx[nodes[4]] -= Fv[ie][7]
        CUDA.@atomic Fx[nodes[5]] -= Fv[ie][9]
        CUDA.@atomic Fx[nodes[6]] -= Fv[ie][11]
        CUDA.@atomic Fx[nodes[7]] -= Fv[ie][13]
        CUDA.@atomic Fy[nodes[1]] -= Fv[ie][2]
        CUDA.@atomic Fy[nodes[2]] -= Fv[ie][4]
        CUDA.@atomic Fy[nodes[3]] -= Fv[ie][6]
        CUDA.@atomic Fy[nodes[4]] -= Fv[ie][8]
        CUDA.@atomic Fy[nodes[5]] -= Fv[ie][10]
        CUDA.@atomic Fy[nodes[6]] -= Fv[ie][12]
        CUDA.@atomic Fy[nodes[7]] -= Fv[ie][14]
                     FP[ie]       -= Fp[ie][1]
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

#----------------------------------------------------------#
function ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX ) # Adapted from MILAMIN_1.0.1
    ndof         = 2*mesh.nnel
    npel         = mesh.npel
    nip          = length(ipw)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    Q_all        = zeros(mesh.nel, ndof, npel)
    Mi_all       = zeros(mesh.nel, npel, npel)
    b            = zeros(mesh.nel, ndof)
    K_ele        = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, npel)
    M_ele        = zeros(npel,npel)
    M_inv        = zeros(npel,npel)
    b_ele        = zeros(ndof)
    B            = zeros(ndof,3)
    m            = [ 1.0; 1.0; 0.0]
    Dev          = [ 4/3 -2/3  0.0;
                    -2/3  4/3  0.0;
                     0.0  0.0  1.0]
    Pr  = ones(npel,npel)
    Pb = ones(npel)
    PF = 1e3*maximum(mesh.ke)
    
    # Element loop
    @inbounds for e = 1:mesh.nel
        nodes   = mesh.e2n[e,:]
        x       = [mesh.xn[nodes] mesh.yn[nodes]]  
        ke      = mesh.ke[e]
        J       = zeros(2,2)
        invJ    = zeros(2,2)
        K_ele  .= 0.0
        Q_ele  .= 0.0
        M_ele  .= 0.0
        b_ele  .= 0.0
        M_inv  .= 0.0
        if npel==3  Pr[2:3,:] .= (x[1:3,:])' end

        # Integration loop
        for ip=1:nip

            if npel==3 
                Ni       = N[ip,:,1]
                Pb[2:3] .= x'*Ni 
                Pi       = Pr\Pb
            end

            dNdXi     = dNdX[ip,:,:]
            J        .= x'*dNdXi
            detJ      = J[1,1]*J[2,2] - J[1,2]*J[2,1]
            invJ[1,1] = +J[2,2] / detJ
            invJ[1,2] = -J[1,2] / detJ
            invJ[2,1] = -J[2,1] / detJ
            invJ[2,2] = +J[1,1] / detJ
            dNdx      = dNdXi*invJ
            B[1:2:end,1] .= dNdx[:,1]
            B[2:2:end,2] .= dNdx[:,2]
            B[1:2:end,3] .= dNdx[:,2]
            B[2:2:end,3] .= dNdx[:,1]
            Bvol          = dNdx'
            K_ele .+= ipw[ip] .* detJ .* ke  .* (B*Dev*B')
            if npel==3 
                Q_ele   .-= ipw[ip] .* detJ .* (Bvol[:]*Pi') 
                M_ele   .+= ipw[ip] .* detJ .* Pi*Pi'
            elseif npel==1 
                Q_ele   .-= ipw[ip] .* detJ .* 1.0 .* (B*m*npel') 
            end
            # b_ele   .+= ipw[ip] .* detJ .* se[e] .* N[ip,:] 
        end
        if npel==3 M_inv .= inv(M_ele) end
        # if npel==3 K_ele .+=  PF.*(Q_ele*M_inv*Q_ele') end
        if npel==3 Mi_all[e,:,:] .= M_inv end
        K_all[e,:,:]  .= K_ele
        Q_all[e,:,:]  .= Q_ele
        # b[e,:]       .= b_ele
    end
    return K_all, Q_all, Mi_all, b
end 

#----------------------------------------------------------#

function SparseAssembly( K_all, Q_all, Mi_all, mesh, Vx, Vy )

    println("Assembly")
    ndof = mesh.nn*2
    npel = mesh.npel
    rhs  = zeros(ndof+mesh.nel*npel)
    I_K  = Int64[]
    J_K  = Int64[]
    V_K  = Float64[]
    I_Q  = Int64[]
    J_Q  = Int64[]
    V_Q  = Float64[]
    I_Qt = Int64[]
    J_Qt = Int64[]
    V_Qt = Float64[]

    # Assembly of global sparse matrix
    @inbounds for e=1:mesh.nel
        nodes   = mesh.e2n[e,:]
        nodesVx = mesh.e2n[e,:]
        nodesVy = nodesVx .+ mesh.nn 
        nodesP  = e
        jj = 1
        for j=1:mesh.nnel

            # Q: ∇ operator: BC for V equations
            if mesh.bcn[nodes[j]] != 1
                for i=1:npel
                    push!(I_Q, nodesVx[j]); push!(J_Q, nodesP+(i-1)*mesh.nel); push!(V_Q, Q_all[e,jj  ,i])
                    push!(I_Q, nodesVy[j]); push!(J_Q, nodesP+(i-1)*mesh.nel); push!(V_Q, Q_all[e,jj+1,i])
                end
            end

            # Qt: ∇⋅ operator: no BC for Pr equations
            for i=1:npel
                push!(J_Qt, nodesVx[j]); push!(I_Qt, nodesP+(i-1)*mesh.nel); push!(V_Qt, Q_all[e,jj  ,i])
                push!(J_Qt, nodesVy[j]); push!(I_Qt, nodesP+(i-1)*mesh.nel); push!(V_Qt, Q_all[e,jj+1,i])
            end

            if mesh.bcn[nodes[j]] != 1
                # If not Dirichlet, add connection
                ii = 1
                for i=1:mesh.nnel
                    if mesh.bcn[nodes[i]] != 1 # Important to keep matrix symmetric
                        push!(I_K, nodesVx[j]); push!(J_K, nodesVx[i]); push!(V_K, K_all[e,jj,  ii  ])
                        push!(I_K, nodesVx[j]); push!(J_K, nodesVy[i]); push!(V_K, K_all[e,jj,  ii+1])
                        push!(I_K, nodesVy[j]); push!(J_K, nodesVx[i]); push!(V_K, K_all[e,jj+1,ii  ])
                        push!(I_K, nodesVy[j]); push!(J_K, nodesVy[i]); push!(V_K, K_all[e,jj+1,ii+1])
                    else
                        rhs[nodesVx[j]] -= K_all[e,jj  ,ii  ]*Vx[nodes[i]]
                        rhs[nodesVx[j]] -= K_all[e,jj  ,ii+1]*Vy[nodes[i]]
                        rhs[nodesVy[j]] -= K_all[e,jj+1,ii  ]*Vx[nodes[i]]
                        rhs[nodesVy[j]] -= K_all[e,jj+1,ii+1]*Vy[nodes[i]]
                    end
                    ii+=2
                end
                # rhs[nodes[j]] += b[e,j]
            else
                # Deal with Dirichlet: set one on diagonal and value and RHS
                push!(I_K, nodesVx[j]); push!(J_K, nodesVx[j]); push!(V_K, 1.0)
                push!(I_K, nodesVy[j]); push!(J_K, nodesVy[j]); push!(V_K, 1.0)
                rhs[nodesVx[j]] += Vx[nodes[j]]
                rhs[nodesVy[j]] += Vy[nodes[j]]
            end
            jj+=2
        end 
    end
    K  = sparse(I_K,  J_K,  V_K, mesh.nn*2, mesh.nn*2)
    Q  = sparse(I_Q,  J_Q,  V_Q, ndof, mesh.nel*npel)
    Qt = sparse(I_Qt, J_Qt, V_Qt, mesh.nel*npel, ndof)
    M0 = sparse(Int64[], Int64[], Float64[], mesh.nel*npel, mesh.nel*npel)
    M  = [K Q; Qt M0]
    return M, rhs, K, Q, Qt, M0
end

#----------------------------------------------------------#

function DirectSolveFEM!( M, K, Q, Qt, M0, rhs, Vx, Vy, Pr, mesh, b)
    println("Direct solve")
    ndof       = mesh.nn*2
    npel       = mesh.npel
    sol        = zeros(ndof+mesh.nel*npel)
    M[end,:]  .= 0.0 # add one Dirichlet constraint on pressure
    M[end,end] = 1.0
    rhs[end]   = 0.0
    sol       .= M\rhs
    Vx        .= sol[1:length(Vx)] 
    Vy        .= sol[(length(Vx)+1):(length(Vx)+length(Vy))]
    println(size( Pr))
    println(( mesh.nel*npel))
    Pr         .= sol[(2*mesh.nn+1):end]
    return nothing
end

#----------------------------------------------------------#

function main(n, nnel, npel, nip, θ, ΔτV, ΔτP)
    println("\n******** FEM Stokes ********")
    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    nx, ny     = Int16(n*30), Int16(n*30)
    R          = 1.0
    inclusion  = 1
    η          = (1.0, 5.0)
    # Element data
    ipx, ipw   = IntegrationTriangle(nip)
    N, dNdX    = ShapeFunctions(ipx, nip, nnel)
    # Generate mesh
    mesh = MakeTriangleMesh(nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R; nnel, npel)
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    # Initial condition
    mesh.ke[mesh.phase.==1] .= η[1]
    mesh.ke[mesh.phase.==2] .= η[2]
    Vx = zeros(mesh.nn)       # Solution on nodes
    Vy = zeros(mesh.nn)       # Solution on nodes
    Pr = zeros(mesh.nel*npel) # Solution on elements
    se = zeros(mesh.nel)      # Source term on elements
    Pa = zeros(mesh.nel)      # Solution on elements
    # Intial guess
    for in = 1:mesh.nn
        if mesh.bcn[in]==1
            x, y = mesh.xn[in], mesh.yn[in]
            vx, vy, p = EvalAnalDani(x, y, R, η[1], η[2])
            Vx[in] = vx
            Vy[in] = vy
        end
    end
    for ie = 1:mesh.nel
        x, y = mesh.xc[ie], mesh.yc[ie]
        vx, vy, p = EvalAnalDani(x, y, R, η[1], η[2])
        Pa[ie] = p
    end
    K_all, Q_all, Mi_all, b = ElementAssemblyLoopFEM(se, mesh, ipw, N, dNdX)
    # Solve
    if USE_DIRECT
        @time M, rhs, K, Q, Qt, M0 = SparseAssembly(K_all, Q_all, Mi_all, mesh, Vx, Vy)
        @time DirectSolveFEM!(M, K, Q, Qt, M0, rhs, Vx, Vy, Pr, mesh, b)
    else
        nout    = 1000
        iterMax = 2e4
        ϵ_PT    = 1e-7
        ΔVxΔτ   = zeros(mesh.nn)
        ΔVyΔτ   = zeros(mesh.nn)
        ΔVxΔτ0  = zeros(mesh.nn)
        ΔVyΔτ0  = zeros(mesh.nn)
        ΔPΔτ    = zeros(mesh.nel*npel)
        V_ele   = zeros(mesh.nnel*2)
        if USE_GPU
        nel = mesh.nel; nn = mesh.nn
        npel=mesh.npel; nnel=mesh.nnel
        threads = 256
        blocks  = nel÷threads + 1
        e2n     = CuArray(mesh.e2n) #(nel, nnel)
        bcn     = CuArray(mesh.bcn) #nn
        ΔVxΔτ   = CuArray(ΔVxΔτ)
        ΔVyΔτ   = CuArray(ΔVyΔτ)
        ΔVxΔτ0  = CuArray(ΔVxΔτ0)
        ΔVyΔτ0  = CuArray(ΔVyΔτ0)
        ΔPΔτ    = CuArray(ΔPΔτ)
        Vx      = CuArray(Vx) #nn
        Vy      = CuArray(Vy) #nn
        Pr      = CuArray(Pr ) #nel*npel
        se      = CuArray(se) #nel
        K_all   = CuArray(K_all) #(nel, nnel*2, nnel*2)
        Q_all   = CuArray(Q_all) #(nel, nnel*2, 1)    
        # local cell array size
        dims_nel = (nel)    
        celldims_K_loc = (nnel*2, nnel*2)
        celldims_Q_loc = (nnel*2, 1)
        celldims_V_loc = (nnel*2, 1)
        celldims_P_loc = (npel  , 1)
        celldims_e2n_l = (nnel  , 1)
        # init FEM tmp CellArrays
        K_loc  = SMatrix{celldims_K_loc..., Float64, prod(celldims_K_loc)}
        K      = CuCellArray{K_loc}(undef, dims_nel)
        Q_loc  = SMatrix{celldims_Q_loc..., Float64, prod(celldims_Q_loc)}
        Q      = CuCellArray{Q_loc}(undef, dims_nel)
        V_loc  = SMatrix{celldims_V_loc..., Float64, prod(celldims_V_loc)}
        V      = CuCellArray{V_loc}(undef, dims_nel)
        P_loc  = SMatrix{celldims_P_loc..., Float64, prod(celldims_P_loc)}
        P      = CuCellArray{P_loc}(undef, dims_nel)
        Fv_loc = SMatrix{celldims_V_loc..., Float64, prod(celldims_V_loc)}
        Fv     = CuCellArray{Fv_loc}(undef, dims_nel)
        Fp_loc = SMatrix{celldims_P_loc..., Float64, prod(celldims_P_loc)}
        Fp     = CuCellArray{Fp_loc}(undef, dims_nel)
        e2n_l  = SMatrix{celldims_e2n_l..., Int64, prod(celldims_e2n_l)}
        E2N    = CuCellArray{e2n_l}(undef, dims_nel)

        @cuda blocks=blocks threads=threads init_CellArrays!(K,Q,E2N,K_all,Q_all,e2n,nel); synchronize()
        end
        # PT loop
        local iter = 0; success = 0
        @time while (iter<iterMax)
            iter += 1
            ΔVxΔτ0 .= ΔVxΔτ
            ΔVyΔτ0 .= ΔVyΔτ
            # V_ele  .= 0.0
            ΔVxΔτ  .= 0.0
            ΔVyΔτ  .= 0.0
            ΔPΔτ   .= 0.0

            # ResidualStokesElementalSerialFEM!(ΔVxΔτ, ΔVyΔτ, ΔPΔτ, V_ele, Vx, Vy, Pr, K_all, Q_all, e2n, bcn, nnel, npel, nel)

            @cuda blocks=blocks threads=threads my_gpu_fun_0!(V,P,Vx,Vy,Pr,E2N,nel); synchronize()
            @cuda blocks=blocks threads=threads my_gpu_fun_1!(Fv,Fp,K,Q,V,P,nel); synchronize()
            @cuda blocks=blocks threads=threads my_gpu_fun_2!(ΔVxΔτ,ΔVyΔτ,ΔPΔτ,Fv,Fp,E2N,nel); synchronize()

            # to be added to kernel
            ΔVxΔτ[bcn.==1] .= 0.0
            ΔVyΔτ[bcn.==1] .= 0.0

            ΔVxΔτ  .= (1.0 - θ).*ΔVxΔτ0 .+ ΔVxΔτ 
            ΔVyΔτ  .= (1.0 - θ).*ΔVyΔτ0 .+ ΔVyΔτ
            Vx    .+= ΔτV .* ΔVxΔτ
            Vy    .+= ΔτV .* ΔVyΔτ
            Pr    .+= ΔτP .* ΔPΔτ
            if iter % nout == 0 || iter==1
                errVx = norm(ΔVxΔτ)/sqrt(length(ΔVxΔτ))
                errVy = norm(ΔVyΔτ)/sqrt(length(ΔVyΔτ))
                errP  = norm(ΔPΔτ)/sqrt(length(ΔPΔτ))
                @printf("PT Iter. %05d:\n", iter)
                @printf("  ||Fx|| = %3.3e\n", errVx)
                @printf("  ||Fy|| = %3.3e\n", errVy)
                @printf("  ||Fp|| = %3.3e\n", errP )
                err = max(errVx, errVy, errP)
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
    end
    # Postprocessing
    # Vxe = zeros(mesh.nel)
    # Vye = zeros(mesh.nel)
    # Pe  = zeros(mesh.nel)
    # for ie=1:mesh.nel
    #     for in=1:mesh.nnel
    #         Vxe[ie] += 1.0/mesh.nnel * Array(Vx)[mesh.e2n[ie,in]]
    #         Vye[ie] += 1.0/mesh.nnel * Array(Vy)[mesh.e2n[ie,in]]
    #     end
    #     if npel==1 Pe[ie] = Array(Pr)[ie] end
    #     if npel==3 Pe[ie] = 1.0/npel * (Array(Pr)[ie] + Array(Pr)[e+mesh.nel] + Array(Pr)[ie+2*mesh.nel]) end
    # end
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

# main(1, 7, 1, 6, 0.030598470000000003, 0.03666666667,  1.0) # nit = 4000
# main(2, 7, 1, 6, 0.030598470000000003/2, 0.03666666667,  1.0) # nit = 9000
main(4, 7, 1, 6, 0.030598470000000003/3, 0.03666666667,  1.0) # nit = 19000
