const USE_GPU      = false  # Not supported yet 
const USE_DIRECT   = true   # Sparse matrix assembly + direct solver
const USE_NODAL    = false  # Nodal evaluation of residual
const USE_PARALLEL = false  # Parallel residual evaluation
const USE_MAKIE    = true   # Visualisation 
np = 1

using Plots

using Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid

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

# function ResidualPoissonElementalSerialFEM!( F, T, mesh, K_all, b )
#     # Residual
#     F .= 0.0
#     @inbounds for e = 1:mesh.nel
#         nodes      = mesh.e2n[e,:]
#         T_ele      = T[nodes]
#         K_ele      = K_all[e,:,:]
#         f_ele      = K_ele*T_ele .- b[e,:]
#         F[nodes] .-= f_ele[:] # This should be atomic
#     end
#     F[mesh.bcn.==1] .= 0.0
#     return nothing
# end

# #----------------------------------------------------------#

# function ResidualPoissonElementalParallelFEM!( F, F_th, chunks, T, mesh, K_all, b )
#     # Residual
#     F     .= 0.0
#     @sync for e_chunk in chunks
#         @spawn begin
#             tid = threadid()
#             fill!(F_th[tid], 0.0)
#             for e in e_chunk
#                 n_ele              = mesh.e2n[e,:]
#                 T_ele              = T[n_ele]
#                 K_ele              = K_all[e,:,:]
#                 f_ele              = K_ele*T_ele .- b[e,:]
#                 F_th[tid][n_ele] .-= f_ele[:] # This should be atomic
#             end
#         end
#     end
#     F .= reduce(+, F_th)
#     F[mesh.bcn.==1] .= 0.0
#     return nothing
# end


#----------------------------------------------------------#

function ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX ) # Adapted from MILAMIN_1.0.1
    ndof         = 2*mesh.nnel
    nip          = length(ipw)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    Q_all        = zeros(mesh.nel, ndof, np)
    b            = zeros(mesh.nel, ndof)
    K_ele        = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, np)
    b_ele        = zeros(ndof)
    B            = zeros(ndof,3)
    Np           = 1.0
    m            = [1.0;   1.0; 0.0]
    Dev          = [ 4//3 -2//3 0.0;
                    -2//3  4//3 0.0;
                     0.0   0.0  1.0]
    P  = ones(np,np)
    Pb = ones(np)
    
    # Element loop
    for e = 1:mesh.nel #@inbounds
        nodes   = mesh.e2n[e,:]
        x       = [mesh.xn[nodes] mesh.yn[nodes]]    
        ke      = mesh.ke[e]
        J       = zeros(2,2)
        invJ    = zeros(2,2)
        K_ele  .= 0.0
        Q_ele  .= 0.0
        b_ele  .= 0.0

        if np==3  P[2:3,:] .= (x[1:3,:])' end

        # Integration loop
        for ip=1:nip

            if np==3 
                Ni = N[ip,:,1]
                Pb[2:3]     = x'*Ni 
                Pi = P\Pb
            end

            dNdXi     = dNdX[ip,:,:]
            J        .= x'*dNdXi
            detJ      = J[1,1]*J[2,2] - J[1,2]*J[2,1]
            invJ[1,1] = +J[2,2] / detJ
            invJ[1,2] = -J[2,1] / detJ
            invJ[2,1] = -J[1,2] / detJ
            invJ[2,2] = +J[1,1] / detJ
            dNdx      = dNdXi*invJ
            B[1:2:end,1] .= dNdx[:,1]
            B[2:2:end,2] .= dNdx[:,2]
            B[1:2:end,3] .= dNdx[:,2]
            B[2:2:end,3] .= dNdx[:,1]
            Bvol         = dNdx'
            # B[1:mesh.nnel,1]     = dNdx[:,1]
            # B[mesh.nnel+1:end,2] = dNdx[:,2]
            # B[1:mesh.nnel,3]     = dNdx[:,1]
            # B[mesh.nnel+1:end,3] = dNdx[:,2]
            K_ele .+= ipw[ip] .* detJ .* ke  .* (B*Dev*B')
            if np==3 Q_ele   .-= ipw[ip] .* detJ .* (Bvol[:]*Pi') end
            if np==1 Q_ele   .-= ipw[ip] .* detJ .* 1.0 .* (B*m*Np') end
            # b_ele   .+= ipw[ip] .* detJ .* se[e] .* N[ip,:] 
        end
        K_all[e,:,:] .= K_ele
        Q_all[e,:,:] .= Q_ele
        # b[e,:]       .= b_ele
    end
    return K_all, Q_all, b
end 

#----------------------------------------------------------#

function DirectSolveFEM!(Vx, Vy, P, mesh, K_all, Q_all, b)

    ndof = length(Vx) + length(Vy)
    K    = sparse(Int64[], Int64[], Float64[], ndof, ndof)
    Q    = sparse(Int64[], Int64[], Float64[], ndof, mesh.nel*np)
    M0   = sparse(Int64[], Int64[], Float64[], mesh.nel*np, mesh.nel*np)
    
    rhs  = zeros(ndof+mesh.nel*np)
    sol  = zeros(ndof+mesh.nel*np)

    # Assembly of global sparse matrix
    for e=1:mesh.nel #@inbounds 
        nodes   = mesh.e2n[e,:]
        nodesVx = mesh.e2n[e,:]
        nodesVy = nodesVx .+ mesh.nn
        nodesP  = e # constant per element
        jj = 1
        for j=1:mesh.nnel
            if mesh.bcn[nodes[j]] == 0
                # If not Dirichlet, add connection
                ii = 1;
                for i=1:mesh.nnel
                    
                    # if mesh.bcn[nodes[i]] == 0 # Important to keep matrix symmetric
                        K[nodesVx[j]        , nodesVx[i]        ] += K_all[e,jj,ii]
                        K[nodesVx[j]        , nodesVy[i]        ] += K_all[e,jj,ii+1]
                        K[nodesVy[j]        , nodesVx[i]        ] += K_all[e,jj+1,ii]
                        K[nodesVy[j]        , nodesVy[i]        ] += K_all[e,jj+1,ii+1]
                
                    # else
                    #     rhs[nodesVx[j]] -= K_all[e,jj,ii]*Vx[nodes[i]]
                    #     rhs[nodesVx[j]] -= K_all[e,jj,ii+1]*Vy[nodes[i]]
                    #     rhs[nodesVy[j]] -= K_all[e,jj+1,i]*Vx[nodes[i]]
                    #     rhs[nodesVy[j]] -= K_all[e,jj+1,ii+1]*Vy[nodes[i]]
                    # end
                    ii = ii + 2;
                end

                for i=1:np
                    Q[nodesVx[j], nodesP + (i-1)*mesh.nel] -= Q_all[e,jj,i]
                    Q[nodesVy[j], nodesP + (i-1)*mesh.nel] -= Q_all[e,jj+1,i]
                end

                # rhs[nodes[j]] += b[e,j]
            else
                # Deal with Dirichlet: set one on diagonal and value and RHS
                rhs[nodesVx[j]] = Vx[nodes[j]]
                rhs[nodesVy[j]] = Vy[nodes[j]]
                K[nodesVx[j], nodesVx[j]] = 1.0
                K[nodesVy[j], nodesVy[j]] = 1.0
            end
            jj = jj +2;
        end 
    end

    println(size(K))
    println(size(Q))
    println(size(M0))
    Qt = Q'

    # Qt[1,:] .= 0.0
    # M0[1,1] = 1.0
    for e=1:mesh.nel
        M0[e,e] = 1e3
    end

    M = [K Q; Qt M0]


    display(spy(M))
    # # Solve using CHOLDMOD
 

    sol .= M\rhs
    # println(size(sol))
    # println(size(sol[mesh.e2n]))
    # println(maximum(mesh.e2n))
    Vx  .= sol[1:length(Vx)] 
    Vy  .= sol[(length(Vx)+1):(length(Vx)+length(Vy))]
    
    println(size( sol[(length(Vx)+length(Vy)+1):(length(Vx)+length(Vy)+mesh.nel)]))
    P   .= sol[(length(Vx)+length(Vy)+1):(length(Vx)+length(Vy)+mesh.nel)]
    # # println(size(sol[(length(Vx)+length(Vy)+1):end]))
    return nothing
end

#----------------------------------------------------------#

function main( n, nnel, θ, Δτ )

    println("\n******** FEM POISSON ********")
    # Create sides of mesh
    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
    nx, ny     = Int16(n*20), Int16(n*20)
    R          = 0.2
    inclusion  = 1
    εBG        = 1.0

    # Element data
    mesh_type  = "UnstructTriangles" 
    nip       = 3
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)
  
    # Generate mesh
    mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R; nnel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)

    mesh.ke[mesh.phase.==1] .= 5

    Vx = zeros(mesh.nn)     # Solution on nodes 
    Vy = zeros(mesh.nn)     # Solution on nodes 
    P  = zeros(mesh.nel)    # Solution on nodes 
    se = zeros(mesh.nel)    # Source term on elements

    # # Intial guess
    # alp = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;
    for in = 1:mesh.nn
        if mesh.bcn[in]==1
            x     = mesh.xn[in]
            y     = mesh.yn[in]
            Vx[in] = -εBG*x
            Vy[in] =  εBG*y
        end
    end
    # display( mesh.bcn[mesh.e2n[1,:]] )
    # display( mesh.xn[mesh.e2n[1,:]] )
    # display( mesh.yn[mesh.e2n[1,:]] )

    # Body force not needed
    # for e=1:mesh.nel
    #     x      = mesh.xc[e]
    #     y      = mesh.yc[e]
    #     Te     = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
    #     se[e] = Te*(-a*alp*cos(a*x + c*y) + b*bet*sin(b*x + d*y))*(a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y)) + Te*(a^2*alp*sin(a*x + c*y) + b^2*bet*cos(b*x + d*y)) + Te*(-alp*c*cos(a*x + c*y) + bet*d*sin(b*x + d*y))*(alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y)) + Te*(alp*c^2*sin(a*x + c*y) + bet*d^2*cos(b*x + d*y))
    # end

    #-----------------------------------------------------------------#
    K_all, Q_all, b = ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX )
    #-----------------------------------------------------------------#
    if USE_DIRECT
        @time DirectSolveFEM!(Vx, Vy, P, mesh, K_all, Q_all, b)
    # else
    #     nout    = 50#1e1
    #     iterMax = 2e3#5e4
    #     ϵ_PT    = 1e-7
    #     # θ       = 0.11428 *1.7
    #     # Δτ      = 0.28 /1.2
    #     # println(minimum(mesh.Γ))
    #     # println(maximum(mesh.Γ))
    #     # println(minimum(mesh.Ω))
    #     # println(maximum(mesh.Ω))
    #     # println("Δτ = ", Δτ)
    #     # Ωe = maximum(mesh.Ω)
    #     # Δx = minimum(mesh.Γ)
    #     # D  = 1.0
    #     # println("Δτ1 = ", Δx^2/(1.1*D) * 1.0/Ωe *2/3)
    #     ΔTΔτ    = zeros(mesh.nn) # Residual
    #     ΔTΔτ_th = [similar(ΔTΔτ) for _ = 1:nthreads()]                   # Parallel: per thread
    #     chunks  = Iterators.partition(1:mesh.nel, mesh.nel ÷ nthreads()) # Parallel: chunks of elements
    #     ΔTΔτ0   = zeros(mesh.nn)

    #     # PT loop
    #     local iter = 0
    #     success = 0
    #     @time while (iter<iterMax)
    #         iter  += 1
    #         ΔTΔτ0 .= ΔTΔτ 
    #         if USE_NODAL
    #             ResidualPoissonNodalFEM!( ΔTΔτ, T, mesh, K_all, b )
    #         else
    #             if USE_PARALLEL==false
    #                 ResidualPoissonElementalSerialFEM!( ΔTΔτ, T, mesh, K_all, b )
    #             else
    #                 ResidualPoissonElementalParallelFEM!( ΔTΔτ, ΔTΔτ_th, chunks, T, mesh, K_all, b )
    #             end
    #         end
    #         ΔTΔτ  .= (1.0 - θ).*ΔTΔτ0 .+ ΔTΔτ 
    #         T    .+= Δτ .* ΔTΔτ
    #         if iter % nout == 0 || iter==1
    #             err = norm(ΔTΔτ)/sqrt(length(ΔTΔτ))
    #             @printf("PT Iter. %05d --- Norm of matrix-free residual: %3.3e\n", iter, err)
    #             if err < ϵ_PT
    #                 print("PT solve converged in")
    #                 success = true
    #                 break
    #             elseif err>1e4
    #                 success = false
    #                 println("exploding !")
    #                 break
    #             elseif isnan(err)
    #                 success = false
    #                 println("NaN !")
    #                 break
    #             end
    #         end
    #     end
    end

    # #-----------------------------------------------------------------#

    Vxe = zeros(mesh.nel)
    Vye = zeros(mesh.nel)
    for e=1:mesh.nel
        for in=1:mesh.nnel
            Vxe[e] += 1.0/mesh.nnel* Vx[mesh.e2n[e,in]]
            Vye[e] += 1.0/mesh.nnel* Vy[mesh.e2n[e,in]]
        end
    end
    @printf("%2.2e %2.2e\n", minimum(Vx), maximum(Vx))
    @printf("%2.2e %2.2e\n", minimum(Vy), maximum(Vy))
    @printf("%2.2e %2.2e\n", minimum(P), maximum(P))

    # # display(Vxe)
    # p1=Plots.plot(mesh.xn[mesh.bcn.==1], mesh.yn[mesh.bcn.==1], markershape=:cross, linewidth=0.0)
    # display(p1)
    if USE_MAKIE
        PlotMakie(mesh, Vxe, xmin, xmax, ymin, ymax)
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


