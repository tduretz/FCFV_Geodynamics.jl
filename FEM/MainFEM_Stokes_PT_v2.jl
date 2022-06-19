const USE_GPU      = false  # Not supported yet 
const USE_DIRECT   = true   # Sparse matrix assembly + direct solver
const USE_NODAL    = false  # Nodal evaluation of residual
const USE_PARALLEL = false  # Parallel residual evaluation
const USE_MAKIE    = true   # Visualisation 
np  = 1
nip = 6

import Plots

using Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid
using MAT

include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../DiscretisationFCFV.jl")
include("IntegrationPoints.jl")

#----------------------------------------------------------#

function StabParam(τ, Γ, Ω, mesh_type, ν) 
    return 0. # Stabilisation is only needed for FCFV
end

# #----------------------------------------------------------#

# function ResidualPoissonNodalFEM!( F, T, mesh, K_all, b )
#     # Residual
#     Threads.@threads for in = 1:mesh.nn
#         F[in] = 0.0
#         if mesh.bcn[in]==0
#             @inbounds for ii=1:length(mesh.n2e[in])
#                 e       = mesh.n2e[in][ii]
#                 nodes   = mesh.e2n[e,:]
#                 T_ele   = T[nodes]  
#                 K       = K_all[e,:,:] 
#                 f       = K*T_ele .- b[e,:]
#                 inode   = mesh.n2e_loc[in][ii]
#                 F[in]  -= f[inode]
#             end
#         end
#     end
#     return nothing
# end

# #----------------------------------------------------------#

function ResidualStokesElementalSerialFEM!( Fx, Fy, Fp, Vx, Vy, P, mesh, K_all, Q_all, b )
    # Residual
    Fx .= 0.0
    Fy .= 0.0
    Fp .= 0.0
    nnel = mesh.nnel
    V_ele = zeros(nnel*2)
    @inbounds for e = 1:mesh.nel
        nodes = mesh.e2n[e,:]
        if np==1 nodesP = [e] end
        if np==3 nodesP = [e; e+mesh.nel; e+2*mesh.nel]  end
        V_ele[1:2:end-1] .= Vx[nodes]
        V_ele[2:2:end]   .= Vy[nodes]
        P_ele      = P[nodesP] 
        K_ele      = K_all[e,:,:]
        Q_ele      = Q_all[e,:,:]
        fv_ele     = K_ele*V_ele .- Q_ele*P_ele # .- b[e,:]
        fp_ele     = Q_ele'*V_ele #.- b[e,:]
        Fx[nodes] .-= fv_ele[1:2:end-1] # This should be atomic
        Fy[nodes] .-= fv_ele[2:2:end]
        if np==1 Fp[nodesP]  -= fp_ele[:]    end
        if np==3 Fp[nodesP] .-= fp_ele[:] end
    end
    Fx[mesh.bcn.==1] .= 0.0
    Fy[mesh.bcn.==1] .= 0.0
    return nothing
end

#----------------------------------------------------------#

function ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX ) # Adapted from MILAMIN_1.0.1
    ndof         = 2*mesh.nnel
    nip          = length(ipw)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    Q_all        = zeros(mesh.nel, ndof, np)
    Mi_all       = zeros(mesh.nel, np, np)
    b            = zeros(mesh.nel, ndof)
    K_ele        = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, np)
    M_ele        = zeros(np,np)
    M_inv        = zeros(np,np)
    b_ele        = zeros(ndof)
    B            = zeros(ndof,3)
    Np           = 1.0
    m            = [ 1.0; 1.0; 0.0]
    Dev          = [ 4/3 -2/3  0.0;
                    -2/3  4/3  0.0;
                     0.0  0.0  1.0]
    P  = ones(np,np)
    Pb = ones(np)
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
        if np==3  P[2:3,:] .= (x[1:3,:])' end

        # Integration loop
        for ip=1:nip

            if np==3 
                Ni       = N[ip,:,1]
                Pb[2:3] .= x'*Ni 
                Pi       = P\Pb
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
            if np==3 
                Q_ele   .-= ipw[ip] .* detJ .* (Bvol[:]*Pi') 
                M_ele   .+= ipw[ip] .* detJ .* Pi*Pi'
            elseif np==1 
                Q_ele   .-= ipw[ip] .* detJ .* 1.0 .* (B*m*Np') 
            end
            # b_ele   .+= ipw[ip] .* detJ .* se[e] .* N[ip,:] 
        end
        if np==3 M_inv .= inv(M_ele) end
        # if np==3 K_ele .+=  PF.*(Q_ele*M_inv*Q_ele') end
        if np==3 Mi_all[e,:,:] .= M_inv end
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
    Q    = sparse(Int64[], Int64[], Float64[], ndof, mesh.nel*np)
    Qt   = sparse(Int64[], Int64[], Float64[], mesh.nel*np, ndof)
    M0   = sparse(Int64[], Int64[], Float64[], mesh.nel*np, mesh.nel*np)
    K    = sparse(Int64[], Int64[], Float64[], mesh.nn*2, mesh.nn*2)
    rhs  = zeros(ndof+mesh.nel*np)
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
                for i=1:np
                    push!(I_Q, nodesVx[j]); push!(J_Q, nodesP+(i-1)*mesh.nel); push!(V_Q, -Q_all[e,jj  ,i])
                    push!(I_Q, nodesVy[j]); push!(J_Q, nodesP+(i-1)*mesh.nel); push!(V_Q, -Q_all[e,jj+1,i])
                end
            end

            # Qt: ∇⋅ operator: no BC for P equations
            for i=1:np
                push!(J_Qt, nodesVx[j]); push!(I_Qt, nodesP+(i-1)*mesh.nel); push!(V_Qt, -Q_all[e,jj  ,i])
                push!(J_Qt, nodesVy[j]); push!(I_Qt, nodesP+(i-1)*mesh.nel); push!(V_Qt, -Q_all[e,jj+1,i])
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
    @time Q  = sparse(I_Q,  J_Q,  V_Q)
    @time K  = sparse(I_K,  J_K,  V_K)
    @time Qt = sparse(I_Qt, J_Qt, V_Qt)
    @time M  = [K Q; Qt M0]
    return M, rhs, K, Q, Qt, M0
end

#----------------------------------------------------------#

function DirectSolveFEM!( M, K, Q, Qt, M0, rhs, Vx, Vy, P, mesh, b)
    println("Direct solve")
    ndof       = mesh.nn*2
    sol        = zeros(ndof+mesh.nel*np)
    M[end,:]  .= 0.0 # add on Dirichlet constraint on pressure
    M[end,end] = 1.0
    rhs[end]   = 0.0
    sol       .= M\rhs
    Vx        .= sol[1:length(Vx)] 
    Vy        .= sol[(length(Vx)+1):(length(Vx)+length(Vy))]
    P         .= sol[ 2*mesh.nn+1:end]
    return nothing
end

#----------------------------------------------------------#

function main( n, nnel, θ, Δτ )

    # file = matopen(string(@__DIR__,"/Milamin.mat"))
    # ELEM2NODE = read(file, "ELEM2NODE")
    # Point_id = read(file, "Point_id")
    # Phases = read(file, "Phases")
    # GCOORD = read(file, "GCOORD")
    # Bc_ind = read(file, "Bc_ind") 
    # Bc_ind = convert(Vector{Int}, Bc_ind[:])
    # Bc_val = read(file, "Bc_val")

    println("\n******** FEM STOKES ********")
    # Create sides of mesh
    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
    nx, ny     = 3,3
    nx, ny     = Int16(n*20), Int16(n*20)
    R          = 0.2
    inclusion  = 1
    εBG        = 1.0

    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)
  
    # Generate mesh
    mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R; nnel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)

    mesh.ke[mesh.phase.==1] .= 5

    Vx = zeros(mesh.nn)     # Solution on nodes 
    Vy = zeros(mesh.nn)     # Solution on nodes 
    P  = zeros(mesh.nel*np) # Solution on nodes 
    se = zeros(mesh.nel)    # Source term on elements

    # Intial guess
    for in = 1:mesh.nn
        if mesh.bcn[in]==1
            x      = mesh.xn[in]
            y      = mesh.yn[in]
            Vx[in] = -εBG*x
            Vy[in] =  εBG*y
        end
    end

    #-----------------------------------------------------------------#
    @time  K_all, Q_all, Mi_all, b = ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX )
    #-----------------------------------------------------------------#
    
    @time M, rhs, K, Q, Qt, M0 = SparseAssembly( K_all, Q_all, Mi_all, mesh, Vx, Vy )
    # if USE_DIRECT
    @time DirectSolveFEM!( M, K, Q, Qt, M0, rhs, Vx, Vy, P, mesh, b )

    Fx = zeros(mesh.nn)
    Fy = zeros(mesh.nn)
    Fp = zeros(mesh.nel*np)
    ResidualStokesElementalSerialFEM!( Fx, Fy, Fp, Vx, Vy, P, mesh, K_all, Q_all, b )

    errVx = norm(Fx)/sqrt(length(Fx))
    errVy = norm(Fx)/sqrt(length(Fy))
    errP  = norm(Fx)/sqrt(length(Fp))
    @printf("Norm of matrix-free residual: %3.3e\n", errVx)
    @printf("Norm of matrix-free residual: %3.3e\n", errVy)
    @printf("Norm of matrix-free residual: %3.3e\n", errP )

    # # else
    # #     nout    = 50#1e1
    # #     iterMax = 2e3#5e4
    # #     ϵ_PT    = 1e-7
    # #     # θ       = 0.11428 *1.7
    # #     # Δτ      = 0.28 /1.2
    # #     # println(minimum(mesh.Γ))
    # #     # println(maximum(mesh.Γ))
    # #     # println(minimum(mesh.Ω))
    # #     # println(maximum(mesh.Ω))
    # #     # println("Δτ = ", Δτ)
    # #     # Ωe = maximum(mesh.Ω)
    # #     # Δx = minimum(mesh.Γ)
    # #     # D  = 1.0
    # #     # println("Δτ1 = ", Δx^2/(1.1*D) * 1.0/Ωe *2/3)
    # #     ΔTΔτ    = zeros(mesh.nn) # Residual
    # #     ΔTΔτ_th = [similar(ΔTΔτ) for _ = 1:nthreads()]                   # Parallel: per thread
    # #     chunks  = Iterators.partition(1:mesh.nel, mesh.nel ÷ nthreads()) # Parallel: chunks of elements
    # #     ΔTΔτ0   = zeros(mesh.nn)

    # #     # PT loop
    # #     local iter = 0
    # #     success = 0
    # #     @time while (iter<iterMax)
    # #         iter  += 1
    # #         ΔTΔτ0 .= ΔTΔτ 
    # #         if USE_NODAL
    # #             ResidualPoissonNodalFEM!( ΔTΔτ, T, mesh, K_all, b )
    # #         else
    # #             if USE_PARALLEL==false
    # #                 ResidualPoissonElementalSerialFEM!( ΔTΔτ, T, mesh, K_all, b )
    # #             else
    # #                 ResidualPoissonElementalParallelFEM!( ΔTΔτ, ΔTΔτ_th, chunks, T, mesh, K_all, b )
    # #             end
    # #         end
    # #         ΔTΔτ  .= (1.0 - θ).*ΔTΔτ0 .+ ΔTΔτ 
    # #         T    .+= Δτ .* ΔTΔτ
    # #         if iter % nout == 0 || iter==1
    # #             err = norm(ΔTΔτ)/sqrt(length(ΔTΔτ))
    # #             @printf("PT Iter. %05d --- Norm of matrix-free residual: %3.3e\n", iter, err)
    # #             if err < ϵ_PT
    # #                 print("PT solve converged in")
    # #                 success = true
    # #                 break
    # #             elseif err>1e4
    # #                 success = false
    # #                 println("exploding !")
    # #                 break
    # #             elseif isnan(err)
    # #                 success = false
    # #                 println("NaN !")
    # #                 break
    # #             end
    # #         end
    # #     end
    # end

    # # #-----------------------------------------------------------------#

    Vxe = zeros(mesh.nel)
    Vye = zeros(mesh.nel)
    Pe  = zeros(mesh.nel)
    
    for e=1:mesh.nel
        for in=1:mesh.nnel
            Vxe[e] += 1.0/mesh.nnel * Vx[mesh.e2n[e,in]]
            Vye[e] += 1.0/mesh.nnel * Vy[mesh.e2n[e,in]]
        end
        if np==1 Pe[e] = P[e] end
        if np==3 Pe[e] = 1.0/np * (P[e] + P[e+mesh.nel] + P[e+2*mesh.nel]) end
    end
    @printf("%2.2e %2.2e\n", minimum(Vx), maximum(Vx))
    @printf("%2.2e %2.2e\n", minimum(Vy), maximum(Vy))
    @printf("%2.2e %2.2e\n", minimum(P), maximum(P))

    # # # display(Vxe)
    # # p1=Plots.plot(mesh.xn[mesh.bcn.==1], mesh.yn[mesh.bcn.==1], markershape=:cross, linewidth=0.0)
    # # display(p1)
    if USE_MAKIE
        PlotMakie(mesh, Vxe, xmin, xmax, ymin, ymax)
        # PlotMakie(mesh, Pe, xmin, xmax, ymin, ymax)
    end
   
end

# Linear elements
# main(1, 3, 0.20398980000000003,        0.23333333333333336) # 150
# main(2, 3, 0.20398980000000003*0.61,   0.23333333333333336) # 250
# main(4, 3, 0.20398980000000003*0.61/2, 0.23333333333333336*0.96) # 500
# main(8, 3, 0.20398980000000003*0.61/4 * 0.98, 0.23333333333333336*0.88) # 1000

# Quadratic elements
# main(1, 6, 0.20398980000000003*0.49,        0.23333333333333336/1.35) # 350
# main(1, 6, 0.20398980000000003*0.49,        0.23333333333333336/1.35) # 350
# main(1, 6, 0.20398980000000003*0.49,        0.23333333333333336/1.35) # 350

main(1, 7, 0.20398980000000003*0.38,        0.23333333333333336/1.44) # 350


