const USE_GPU      = false  # Not supported yet 
const USE_DIRECT   = false  # Sparse matrix assembly + direct solver
const USE_NODAL    = false  # Nodal evaluation of residual
const USE_PARALLEL = false  # Parallel residual evaluation
const USE_MAKIE    = true   # Visualisation 
import Plots
import Statistics:mean

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
        nodes  = mesh.e2n[e,:]
        nodesP = mesh.e2p[e,:]
        V_ele[1:2:end-1] .= Vx[nodes]
        V_ele[2:2:end]   .= Vy[nodes]
        Q_ele      = Q_all[e,:,:]
        fp_ele     = .-Q_ele'*V_ele #.- bp[e,:]
        for p=1:npel
            Fp[e] = - fp_ele[:][p]
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
        nodes  = mesh.e2n[e,:]
        nodesP = mesh.e2p[e,:]
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

function ElementAssemblyLoopFEM( se, mesh, ipx, ipw, N, dNdX ) # Adapted from MILAMIN_1.0.1
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
    m            = [ 1.0; 1.0; 0.0]
    Dev          = [ 4/3 -2/3  0.0;
                    -2/3  4/3  0.0;
                     0.0  0.0  1.0]
    P  = ones(npel,npel)
    Pb = ones(npel)
    Np0, dNdXp   = ShapeFunctions(ipx, nip, 3)

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
                Ni       = N[ip,:,1]
                Pb[2:3] .= x'*Ni 
                Pi       = P\Pb
                Q_ele   .-= ipw[ip] .* detJ .* (Bvol[:]*Pi') 
                # M_ele   .+= ipw[ip] .* detJ .* Pi*Pi' # mass matrix P, not needed sor incompressible
            else
                if npel==1 Np = 1.0        end
                if npel==3 Np = Np0[ip,:,:] end
                Q_ele   .-= ipw[ip] .* detJ .* (B*m*Np') 
            end
            # b_ele   .+= ipw[ip] .* detJ .* se[e] .* N[ip,:] 
        end
        K_all[e,:,:]  .= K_ele
        Q_all[e,:,:]  .= Q_ele
        # b[e,:]       .= b_ele
    end
    return K_all, Q_all, Mi_all, b
end 

#----------------------------------------------------------#

function SparseAssembly( K_all, Q_all, Mi_all, mesh, Vx, Vy, P )

    println("Assembly")
    ndof = mesh.nn*2
    npel = mesh.npel
    rhs  = zeros(ndof+mesh.np)
    I_K,  J_K,  V_K  = Int64[], Int64[], Float64[]
    I_Q,  J_Q,  V_Q  = Int64[], Int64[], Float64[]
    I_Qt, J_Qt, V_Qt = Int64[], Int64[], Float64[] 
    I_M,  J_M,  V_M  = Int64[], Int64[], Float64[]

    # Assembly of global sparse matrix
    @inbounds for e=1:mesh.nel
        nodes   = mesh.e2n[e,:]
        nodesVx = mesh.e2n[e,:]
        nodesVy = nodesVx .+ mesh.nn 
        nodesP  = mesh.e2p[e,:]
        jj = 1
        for j=1:mesh.nnel

            # Q: ∇ operator: BC for V equations
            if mesh.bcn[nodes[j]] != 1
                for i=1:npel
                    push!(I_Q, nodesVx[j]); push!(J_Q, nodesP[i]); push!(V_Q, Q_all[e,jj  ,i])
                    push!(I_Q, nodesVy[j]); push!(J_Q, nodesP[i]); push!(V_Q, Q_all[e,jj+1,i])
                end
            end

            # Qt: ∇⋅ operator: no BC for P 
            for i=1:npel
                push!(J_Qt, nodesVx[j]); push!(I_Qt, nodesP[i]); push!(V_Qt, Q_all[e,jj  ,i])
                push!(J_Qt, nodesVy[j]); push!(I_Qt, nodesP[i]); push!(V_Qt, Q_all[e,jj+1,i])
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
    K  = sparse(I_K,  J_K,  V_K, ndof, ndof)
    Q  = sparse(I_Q,  J_Q,  V_Q, ndof, mesh.np)
    Qt = sparse(I_Qt, J_Qt, V_Qt, mesh.np, ndof)
    M0 = sparse(I_M,  J_M,  V_M, mesh.np, mesh.np)
    M  = [K Q; Qt M0]
    return M, rhs, K, Q, Qt, M0
end

#----------------------------------------------------------#

function DirectSolveFEM!( M, K, Q, Qt, M0, rhs, Vx, Vy, P, mesh, b)
    println("Direct solve")
    ndof       = mesh.nn*2
    sol        = zeros(ndof+mesh.np)
    M[end,:]  .= 0.0 # add one Dirichlet constraint on pressure
    M[end,end] = 1.0
    rhs[end]   = 0.0
    sol       .= M\rhs
    Vx        .= sol[1:length(Vx)] 
    Vy        .= sol[(length(Vx)+1):(length(Vx)+length(Vy))]
    P         .= sol[(ndof+1):end]
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
    η          = [1.0 100.0]  
    
    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)
  
    # Generate mesh
    mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R; nnel, npel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    println("Number of p nodes:  ", mesh.np)

    mesh.ke[mesh.phase.==1] .= η[1]
    mesh.ke[mesh.phase.==2] .= η[2]

    Vx = zeros(mesh.nn)       # Solution on nodes 
    Vy = zeros(mesh.nn)       # Solution on nodes 
    P  = zeros(mesh.np)       # Solution on either elements or nodes 
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
    for in = 1:mesh.nv
        if mesh.bcn[in]==1
            x      = mesh.xn[in]
            y      = mesh.yn[in]
            vx, vy, p = EvalAnalDani( x, y, R, η[1], η[2] )
            P[in] = p
        end
    end

    #-----------------------------------------------------------------#
    @time  K_all, Q_all, Mi_all, b = ElementAssemblyLoopFEM( se, mesh, ipx, ipw, N, dNdX )
    #-----------------------------------------------------------------#
    
    if USE_DIRECT
        @time M, rhs, K, Q, Qt, M0 = SparseAssembly( K_all, Q_all, Mi_all, mesh, Vx, Vy, P )
        @time DirectSolveFEM!( M, K, Q, Qt, M0, rhs, Vx, Vy, P, mesh, b )
    else
        nout    = 1000#1e1
        iterMax = 10e3#3e4
        ϵ_PT    = 5e-7
        ΔVxΔτ = zeros(mesh.nn)
        ΔVyΔτ = zeros(mesh.nn)
        ΔVxΔτ0= zeros(mesh.nn)
        ΔVyΔτ0= zeros(mesh.nn)
        ΔPΔτ  = zeros(mesh.np)

        #-----------------------------------------------#
        # Local Δτ for momentum equations (local Δτ for continuity does not seem to help)
        #-----------------------------------------------#
        ΔτVv  = zeros(mesh.nn)
        ΔτPv  = zeros(mesh.np)
        ηv    = zeros(mesh.nn)
        ηe    = zeros(mesh.nel)
        ηe   .= mesh.ke
        ΔτVv .= ΔτV
        ΔτPv .= ΔτP
        nludo = 1 # more than 1 does not seem to help
        itp   = 0 # only 0 and 3 seem to work

        for iludo=1:nludo
            # Compute nodal viscosities
            for i=1:mesh.nn
                n = 0
                η = 0.0
                for ii=1:length(mesh.n2e[i])
                    e       = mesh.n2e[i][ii]
                    n   += 1
                    if itp==3 η  = max(η, ηe[e]) end # Local maximum
                    if itp==0 η += ηe[e]         end # arithmetic mean
                    if itp==1 η += 1.0/ηe[e]     end # harmonic mean
                    if itp==2 η += log(ηe[e])    end # geometric mean
                end
                w = 1.0/n
                if itp==0 ηv[i] = w*η      end
                if itp==1 ηv[i] = w/η      end
                if itp==2 ηv[i] = exp(η)^w end
                if itp==3 ηv[i] = η        end
            end
            # Compute element viscosities
            for e=1:mesh.nel
                nodes = mesh.e2n[e,:]
                if itp==3 ηe[e] = max(ηv[nodes]...)  end
                if itp==0 ηe[e] = mean(ηv[nodes]) end
            end
        end
        ΔτVv ./= ηv
        #-----------------------------------------------#

        # PT loop
        local iter = 0
        success = 0
        @time while (iter<iterMax)
            iter  += 1
            ΔVxΔτ0 .= ΔVxΔτ 
            ΔVyΔτ0 .= ΔVyΔτ
            if USE_NODAL
                ResidualStokesNodalFEM!( ΔVxΔτ, ΔVyΔτ, ΔPΔτ, Vx, Vy, P, mesh, K_all, Q_all, b )
            else
                ResidualStokesElementalSerialFEM!( ΔVxΔτ, ΔVyΔτ, ΔPΔτ, Vx, Vy, P, mesh, K_all, Q_all, b )
            end
            ΔVxΔτ  .= (1.0 - θ).*ΔVxΔτ0 .+ ΔVxΔτ 
            ΔVyΔτ  .= (1.0 - θ).*ΔVyΔτ0 .+ ΔVyΔτ
            Vx    .+= ΔτVv .* ΔVxΔτ
            Vy    .+= ΔτVv .* ΔVyΔτ
            P     .+= ΔτPv .* ΔPΔτ
            if iter % nout == 0 || iter==1
                errVx = norm(ΔVxΔτ)/sqrt(length(ΔVxΔτ))
                errVy = norm(ΔVyΔτ)/sqrt(length(ΔVyΔτ))
                errP  = norm(ΔPΔτ) /sqrt(length(ΔPΔτ))
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
        for i=1:mesh.nnel
            Vxe[e] += 1.0/mesh.nnel * Vx[mesh.e2n[e,i]]
            Vye[e] += 1.0/mesh.nnel * Vy[mesh.e2n[e,i]]
        end
        for i=1:mesh.npel
            Pe[e] += 1.0/mesh.npel * P[mesh.e2p[e,i]]
        end
    end
    @printf("%2.2e %2.2e\n", minimum(Vx), maximum(Vx))
    @printf("%2.2e %2.2e\n", minimum(Vy), maximum(Vy))
    @printf("%2.2e %2.2e\n", minimum(P),  maximum(P) )
    #-----------------------------------------------------------------#
    if USE_MAKIE
        PlotMakie(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1)
    else
        PlotPyPlot(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1 )
    end
end

main(1, 7, 3, 6, 0.0382, 0.1833, 7.0) # nit = xxxxx
