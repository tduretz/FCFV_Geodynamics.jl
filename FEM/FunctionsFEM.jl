
using SparseArrays

#----------------------------------------------------------#
function StabParam(τ, Γ, Ω, mesh_type, ν) 
    return 0. # Stabilisation is only needed for FCFV
end

#----------------------------------------------------------#
function ResidualStokesNodalFEM!( Fx, Fy, Fp, Vx, Vy, P, mesh, K_all, Q_all, b_all )
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
                b_ele      = b_all[e,:]
                fv_ele     = K_ele*V_ele .+ Q_ele*P_ele .- b_ele
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

function ResidualStokesElementalSerialFEM!( Fx, Fy, Fp, Vx, Vy, P, mesh, K_all, Q_all, b_all )
    # Residual
    Fx   .= 0.0
    Fy   .= 0.0
    Fp   .= 0.0
    nnel  = mesh.nnel
    npel  = mesh.npel
    V_ele = zeros(nnel*2)
    b_ele = zeros(nnel*2)
    @inbounds for e = 1:mesh.nel
        nodes  = mesh.e2n[e,:]
        nodesP = mesh.e2p[e,:]
        V_ele[1:2:end-1] .= Vx[nodes]
        V_ele[2:2:end]   .= Vy[nodes]
        P_ele      = P[nodesP] 
        K_ele      = K_all[e,:,:]
        Q_ele      = Q_all[e,:,:]
        b_ele      = b_all[e,:]
        fv_ele     = K_ele*V_ele .+ Q_ele*P_ele .- b_ele
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

function ElementAssemblyLoopFEM( se, mesh, ipx, ipw, N, dNdX, Vx, Vy, P ) # Adapted from MILAMIN_1.0.1
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = length(ipw)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    Q_all        = zeros(mesh.nel, ndof, npel)
    Mi_all       = zeros(mesh.nel, npel, npel)
    b_all        = zeros(mesh.nel, ndof )
    K_ele        = zeros(ndof, ndof)
    K_ele_bc     = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, npel)
    Q_ele_bc     = zeros(ndof, npel)
    M_ele        = zeros(npel ,npel)
    M_inv        = zeros(npel ,npel)
    b_ele        = zeros(ndof)
    B            = zeros(ndof, 3)
    Bvol         = zeros(ndof)
    x            = zeros(mesh.nnel, 2)
    m            =  [ 1.0; 1.0; 0.0]
    Dev          =  [ 4/3 -2/3  0.0;
                     -2/3  4/3  0.0;
                      0.0  0.0  1.0]
    J            = zeros(2,2)
    invJ         = zeros(2,2)
    P            = ones(npel,npel)
    Pb           = ones(npel)
    dNdx         = zeros(nnel, 2)
    Pi           = zeros(npel)
    nodes        = zeros(Int64,nnel)
    bc           = zeros(Int64,ndof)
    bcv          = zeros(ndof)
    bcp          = ones(npel)
    bc_dir       = zeros(ndof)
    # Element loop
    @inbounds for e = 1:mesh.nel
        nodes  .= mesh.e2n[e,:]
        x      .= [mesh.xn[nodes] mesh.yn[nodes]]  
        K_ele  .= 0.0
        Q_ele  .= 0.0
        M_ele  .= 0.0
        b_ele  .= 0.0
        M_inv  .= 0.0
        if npel==3 && nnel!=4 P[2:3,:] .= (x[1:3,:])' end
        # Deal with BC's
        bc[1:2:end]     .= mesh.bcn[nodes]
        bc[2:2:end]     .= mesh.bcn[nodes]
        bcv             .= 1.0
        bcv[bc.==1]     .= 0.0 
        K_ele_bc        .= bcv*bcv'
        Q_ele_bc        .= bcv*bcp'
        bc_dir[1:2:end] .= Vx[nodes]
        bc_dir[2:2:end] .= Vy[nodes]
        # Integration loop
        for ip=1:nip
            J        .= x'*dNdX[ip,:,:]
            detJ      = J[1,1]*J[2,2] - J[1,2]*J[2,1]
            w         = ipw[ip] * detJ
            invJ[1,1] = +J[2,2] / detJ
            invJ[1,2] = -J[1,2] / detJ
            invJ[2,1] = -J[2,1] / detJ
            invJ[2,2] = +J[1,1] / detJ
            dNdx     .= dNdX[ip,:,:]*invJ
            B[1:2:end,1]  .= dNdx[:,1]
            B[2:2:end,2]  .= dNdx[:,2]
            B[1:2:end,3]  .= dNdx[:,2]
            B[2:2:end,3]  .= dNdx[:,1]
            Bvol[1:2:end] .= dNdx[:,1]
            Bvol[2:2:end] .= dNdx[:,2]
            K_ele        .+= w .* mesh.ke[e]  .* (B*Dev*B')
            if npel==3
                Pb[2:3]  .= x'*N[ip,:,1]
                Pi       .= P\Pb
                Q_ele   .-= w .* (Bvol*Pi') 
                # M_ele   .+= ipw[ip] .* detJ .* Pi*Pi' # mass matrix P, not needed for incompressible
            elseif npel==1  
                Q_ele   .-= w .* (B*m*1.0') 
            end
            b_ele[1:2:end] .+= w .* se[e,1] .* N[ip,:] 
            b_ele[2:2:end] .+= w .* se[e,2] .* N[ip,:]
        end
        K_all[e,:,:]  .= K_ele_bc.*K_ele
        Q_all[e,:,:]  .= Q_ele_bc.*Q_ele
        b_all[e,:]    .= bcv.*b_ele .+ (1.0.-bcv).*bc_dir
    end
    return K_all, Q_all, Mi_all, b_all
end 

#----------------------------------------------------------#

function SparseAssembly( K_all, Q_all, Mi_all, b_all, mesh, Vx, Vy, P )

    println("Assembly")
    ndof = mesh.nn*2
    npel = mesh.npel
    bu   = zeros(ndof)
    bp   = zeros(mesh.np)
    I_K  = Int64[]
    J_K  = Int64[]
    V_K  = Float64[]
    I_Q  = Int64[]
    J_Q  = Int64[]
    V_Q  = Float64[]
    I_Qt = Int64[]
    J_Qt = Int64[]
    V_Qt = Float64[]
    I_M  = Int64[]
    J_M  = Int64[]
    V_M  = Float64[]

    # Assembly of global sparse matrix
    @inbounds for e=1:mesh.nel
        nodes   = mesh.e2n[e,:]
        nodesVx = mesh.e2n[e,:]
        nodesVy = nodesVx .+ mesh.nn 
        nodesP  = mesh.e2p[e,:]
        jj = 1
        for j=1:mesh.nnel

            bc = mesh.bcn[nodes[j]]
            if bc==0 || bc==2 
                bcxj = 0
                bcyj = 0
            end
            if bc==1 
                bcxj = 1
                bcyj = 1
            end
            if bc==3 
                bcxj = 1
                bcyj = 0
            end
            if bc==4 
                bcxj = 0
                bcyj = 1
            end

            # Q: ∇ operator: BC for V equations
            for i=1:npel
                if bcxj==0
                    push!(I_Q, nodesVx[j]); push!(J_Q, nodesP[i]); push!(V_Q, Q_all[e,jj  ,i])
                end
                if bcyj==0
                    push!(I_Q, nodesVy[j]); push!(J_Q, nodesP[i]); push!(V_Q, Q_all[e,jj+1,i])
                end
            end

            # Qt: ∇⋅ operator: no BC for P 
            # if mesh.bcn[nodes[j]] != 1
            for i=1:npel
                push!(J_Qt, nodesVx[j]); push!(I_Qt, nodesP[i]); push!(V_Qt, Q_all[e,jj  ,i])
                push!(J_Qt, nodesVy[j]); push!(I_Qt, nodesP[i]); push!(V_Qt, Q_all[e,jj+1,i])
            end
            # end

            if bcxj!=1
                 # If not Dirichlet, add connection
                 ii = 1
                 for i=1:mesh.nnel
                     if mesh.bcn[nodes[i]] != 1 # Important to keep matrix symmetric
                         push!(I_K, nodesVx[j]); push!(J_K, nodesVx[i]); push!(V_K, K_all[e,jj,  ii  ])
                         push!(I_K, nodesVx[j]); push!(J_K, nodesVy[i]); push!(V_K, K_all[e,jj,  ii+1])
                     else
                         bu[nodesVx[j]] -= K_all[e,jj  ,ii  ]*Vx[nodes[i]]
                         bu[nodesVx[j]] -= K_all[e,jj  ,ii+1]*Vy[nodes[i]]
                     end
                     ii+=2
                 end
                 bu[nodesVx[j]] += b_all[e,jj]
             else
                 # Deal with Dirichlet: set one on diagonal and value and RHS
                 push!(I_K, nodesVx[j]); push!(J_K, nodesVx[j]); push!(V_K, 1.0)
                 bu[nodesVx[j]] += Vx[nodes[j]]
             end

            if bcyj!=1
                # If not Dirichlet, add connection
                ii = 1
                for i=1:mesh.nnel
                     if mesh.bcn[nodes[i]] != 1 # Important to keep matrix symmetric
                         push!(I_K, nodesVy[j]); push!(J_K, nodesVx[i]); push!(V_K, K_all[e,jj+1,ii  ])
                         push!(I_K, nodesVy[j]); push!(J_K, nodesVy[i]); push!(V_K, K_all[e,jj+1,ii+1])
                     else
                         bu[nodesVy[j]] -= K_all[e,jj+1,ii  ]*Vx[nodes[i]]
                         bu[nodesVy[j]] -= K_all[e,jj+1,ii+1]*Vy[nodes[i]]
                     end
                     ii+=2
                 end
                 bu[nodesVy[j]] += b_all[e,jj+1]
             else
                 # Deal with Dirichlet: set one on diagonal and value and RHS
                 push!(I_K, nodesVy[j]); push!(J_K, nodesVy[j]); push!(V_K, 1.0)
                 bu[nodesVy[j]] += Vy[nodes[j]]
             end
            jj+=2
        end 
    end
    K  = sparse(I_K,  J_K,  V_K, ndof, ndof)
    Q  = sparse(I_Q,  J_Q,  V_Q, ndof, mesh.np)
    Qt = sparse(I_Qt, J_Qt, V_Qt, mesh.np, ndof)
    M0 = sparse(I_M,  J_M,  V_M, mesh.np, mesh.np)
    M  = [K Q; Qt M0]
    return M, bu, bp, K, Q, Qt, M0
end

#----------------------------------------------------------#