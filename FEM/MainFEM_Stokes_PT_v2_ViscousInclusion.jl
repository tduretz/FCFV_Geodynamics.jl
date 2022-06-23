const USE_GPU      = false  # Not supported yet 
const USE_DIRECT   = true   # Sparse matrix assembly + direct solver
const USE_NODAL    = false  # Nodal evaluation of residual
const USE_PARALLEL = false  # Parallel residual evaluation
const USE_MAKIE    = true   # Visualisation 
import Plots

using Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid
using MAT

include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../DiscretisationFCFV.jl")
include("../EvalAnalDani.jl")
include("IntegrationPoints.jl")

#----------------------------------------------------------#

function StabParam(τ, Γ, Ω, mesh_type, ν) 
    return 0. # Stabilisation is only needed for FCFV
end

#----------------------------------------------------------#

function ResidualStokesNodalFEM!( Fx, Fy, Fp, Vx, Vy, P, mesh, K_all, Q_all, b )
    # Residual
    Fx   .= 0.0
    Fy   .= 0.0
    Fp   .= 0.0
    nnel  = mesh.nnel
    npel  = mesh.npel
    ###################################### VELOCITY
    # Loop over nodes and elements connected to each node to avoid race condition - quite horrible
    Threads.@threads for in = 1:mesh.nn
        Fx[in] = 0.0
        Fy[in] = 0.0
        if mesh.bcn[in]==0
            for ii=1:length(mesh.n2e[in])
                e       = mesh.n2e[in][ii]
                nodes   = mesh.e2n[e,:]
                if npel==1 nodesP = [e] end
                if npel==3 nodesP = [e; e+mesh.nel; e+2*mesh.nel]  end
                V_ele             = zeros(nnel*2)
                V_ele[1:2:end-1] .= Vx[nodes]
                V_ele[2:2:end]   .= Vy[nodes]  
                P_ele      = P[nodesP] 
                K_ele      = K_all[e,:,:]
                Q_ele      = Q_all[e,:,:]
                fv_ele     = K_ele*V_ele .+ Q_ele*P_ele # .- bu[e,:]
                inode      = mesh.n2e_loc[in][ii]
                Fx[in]    -= fv_ele[2*inode-1]
                Fy[in]    -= fv_ele[2*inode]
            end
        end
    end
    ###################################### PRESSURE
    # Pressure is discontinuous across elements, the residual can be evaluated per element without race condition
    V_ele = zeros(nnel*2)
    @inbounds for e = 1:mesh.nel
        Fp[e] = 0.0
        nodes = mesh.e2n[e,:]
        if npel==1 nodesP = [e] end
        if npel==3 nodesP = [e; e+mesh.nel; e+2*mesh.nel]  end
        V_ele[1:2:end-1] .= Vx[nodes]
        V_ele[2:2:end]   .= Vy[nodes]
        Q_ele      = Q_all[e,:,:]
        fp_ele     = .-Q_ele'*V_ele #.- bp[e,:]
        for p=1:npel
            Fp[e+(p-1)*mesh.nel] = - fp_ele[:][p]
        end
    end
    return nothing
end

#----------------------------------------------------------#

function ResidualStokesElementalSerialFEM!( Fx, Fy, Fp, Vx, Vy, P, mesh, K_all, Q_all, b )
    # Residual
    Fx   .= 0.0
    Fy   .= 0.0
    Fp   .= 0.0
    nnel  = mesh.nnel
    npel  = mesh.npel
    V_ele = zeros(nnel*2)
    @inbounds for e = 1:mesh.nel
        nodes = mesh.e2n[e,:]
        if npel==1 nodesP = [e] end
        if npel==3 nodesP = [e; e+mesh.nel; e+2*mesh.nel]  end
        V_ele[1:2:end-1] .= Vx[nodes]
        V_ele[2:2:end]   .= Vy[nodes]
        P_ele      = P[nodesP] 
        K_ele      = K_all[e,:,:]
        Q_ele      = Q_all[e,:,:]
        fv_ele     = K_ele*V_ele .+ Q_ele*P_ele # .- bu[e,:]
        fp_ele     = .-Q_ele'*V_ele #.- bp[e,:]
        Fx[nodes] .-= fv_ele[1:2:end-1] # This should be atomic
        Fy[nodes] .-= fv_ele[2:2:end]
        if npel==1 Fp[nodesP]  -= fp_ele[:] end
        if npel==3 Fp[nodesP] .-= fp_ele[:] end
    end
    Fx[mesh.bcn.==1] .= 0.0
    Fy[mesh.bcn.==1] .= 0.0
    return nothing
end

#----------------------------------------------------------#

function ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX ) # Adapted from MILAMIN_1.0.1
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
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
    Np           = zeros(ndof,3)
    m            = [ 1.0; 1.0; 0.0]
    Dev          = [ 4/3 -2/3  0.0;
                    -2/3  4/3  0.0;
                     0.0  0.0  1.0]
    P  = ones(npel,npel)
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
        if npel==3 && nnel!=4 P[2:3,:] .= (x[1:3,:])' end

        # Integration loop
        for ip=1:nip

            if npel==3 && nnel!=4 
                Ni       = N[ip,:,1]
                Pb[2:3] .= x'*Ni 
                Pi       = P\Pb
            else
                Np[1:2:end,1] .= N[ip,:,1:3]
                Np[2:2:end,1] .= N[ip,:,1:3]
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
            if npel==3 && nnel!=4 
                Q_ele   .-= ipw[ip] .* detJ .* (Bvol[:]*Pi') 
                M_ele   .+= ipw[ip] .* detJ .* Pi*Pi'
            elseif npel==1 && nnel!=4
                Q_ele   .-= ipw[ip] .* detJ .* 1.0 .* (B*m*npel') 
            else
                println(size(Q_ele))
                println(size(B*Np''))
                Q_ele   .-= ipw[ip] .* detJ .* 1.0 .* (B*Np') 
            end
            # b_ele   .+= ipw[ip] .* detJ .* se[e] .* N[ip,:] 
        end
        if npel==3 && nnel!=4 
            M_inv .= inv(M_ele) 
            # if npel==3 K_ele .+=  PF.*(Q_ele*M_inv*Q_ele') end
            Mi_all[e,:,:] .= M_inv 
        end
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

            # Qt: ∇⋅ operator: no BC for P equations
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

function DirectSolveFEM!( M, K, Q, Qt, M0, rhs, Vx, Vy, P, mesh, b)
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
    println(size( P))
    println(( mesh.nel*npel))
    P         .= sol[(2*mesh.nn+1):end]
    return nothing
end

#----------------------------------------------------------#

function main( n, nnel, npel, nip, θ, ΔτV, ΔτP )

    println("\n******** FEM STOKES ********")
    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    nx, ny     = Int16(n*30), Int16(n*30)
    R          = 1.0
    inclusion  = 1
    εBG        = 1.0
    η          = [1.0 5.0]  
    
    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)
  
    # Generate mesh
    mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R; nnel, npel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)

    mesh.ke[mesh.phase.==1] .= η[1]
    mesh.ke[mesh.phase.==2] .= η[2]

    Vx = zeros(mesh.nn)       # Solution on nodes 
    Vy = zeros(mesh.nn)       # Solution on nodes 
    P  = zeros(mesh.nel*npel) # Solution on elements 
    se = zeros(mesh.nel)      # Source term on elements
    Pa = zeros(mesh.nel)      # Solution on elements 

    # Intial guess
    for in = 1:mesh.nn
        if mesh.bcn[in]==1
            x      = mesh.xn[in]
            y      = mesh.yn[in]
            vx, vy = EvalAnalDani( x, y, R, η[1], η[2] )
            Vx[in] = vx
            Vy[in] = vy
        end
    end
    for e = 1:mesh.nel
        x      = mesh.xc[e]
        y      = mesh.yc[e]
        vx, vy, p = EvalAnalDani( x, y, R, η[1], η[2] )
        Pa[e] = p
    end

    #-----------------------------------------------------------------#
    @time  K_all, Q_all, Mi_all, b = ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX )
    #-----------------------------------------------------------------#
    
    if USE_DIRECT
        @time M, rhs, K, Q, Qt, M0 = SparseAssembly( K_all, Q_all, Mi_all, mesh, Vx, Vy )
        @time DirectSolveFEM!( M, K, Q, Qt, M0, rhs, Vx, Vy, P, mesh, b )
    else
        nout    = 1000#1e1
        iterMax = 2e4#5e4
        ϵ_PT    = 1e-7
        # θ       = 0.11428 *1.7
        # Δτ      = 0.28 /1.2
        ΔVxΔτ = zeros(mesh.nn)
        ΔVyΔτ = zeros(mesh.nn)
        ΔVxΔτ0= zeros(mesh.nn)
        ΔVyΔτ0= zeros(mesh.nn)
        ΔPΔτ  = zeros(mesh.nel*npel)
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

        # PT loop
        local iter = 0
        success = 0
        @time while (iter<iterMax)
            iter  += 1
            ΔVxΔτ0 .= ΔVxΔτ 
            ΔVyΔτ0 .= ΔVyΔτ
            # if USE_NODAL
                ResidualStokesNodalFEM!( ΔVxΔτ, ΔVyΔτ, ΔPΔτ, Vx, Vy, P, mesh, K_all, Q_all, b )
            # else
                ResidualStokesElementalSerialFEM!( ΔVxΔτ, ΔVyΔτ, ΔPΔτ, Vx, Vy, P, mesh, K_all, Q_all, b )
            # end
            ΔVxΔτ  .= (1.0 - θ).*ΔVxΔτ0 .+ ΔVxΔτ 
            ΔVyΔτ  .= (1.0 - θ).*ΔVyΔτ0 .+ ΔVyΔτ
            Vx    .+= ΔτV .* ΔVxΔτ
            Vy    .+= ΔτV .* ΔVyΔτ
            P     .+= ΔτP .* ΔPΔτ
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
                    print("PT solve converged in ")
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

    #-----------------------------------------------------------------#
    Vxe = zeros(mesh.nel)
    Vye = zeros(mesh.nel)
    Pe  = zeros(mesh.nel)
    for e=1:mesh.nel
        for in=1:mesh.nnel
            Vxe[e] += 1.0/mesh.nnel * Vx[mesh.e2n[e,in]]
            Vye[e] += 1.0/mesh.nnel * Vy[mesh.e2n[e,in]]
        end
        if npel==1 Pe[e] = P[e] end
        if npel==3 Pe[e] = 1.0/npel * (P[e] + P[e+mesh.nel] + P[e+2*mesh.nel]) end
    end
    @printf("%2.2e %2.2e\n", minimum(Vx), maximum(Vx))
    @printf("%2.2e %2.2e\n", minimum(Vy), maximum(Vy))
    @printf("%2.2e %2.2e\n", minimum(P),  maximum(P) )
    #-----------------------------------------------------------------#
    if USE_MAKIE
        # PlotMakie(mesh, Vxe, xmin, xmax, ymin, ymax; cmap=:jet1)
        PlotMakie(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1 )
    else
        PlotPyPlot(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1 )
    end
end

# main(1, 7, 1, 6, 0.030598470000000003, 0.03666666667,  1.0) # nit = 4000
# main(2, 7, 1, 6, 0.030598470000000003/2, 0.03666666667,  1.0) # nit = 9000
main(1, 7, 3, 6, 0.030598470000000003, 0.03666666667,  1.0) # nit = 4000



