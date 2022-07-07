
using SparseArrays

psize(a) = println(size(a))

#----------------------------------------------------------#
function StabParam(τ, Γ, Ω, mesh_type, ν) 
    return 0. # Stabilisation is only needed for FCFV
end

#----------------------------------------------------------#

function ComputeStressFEM!( εkk, εxx, εyy, εxy, τxx, τyy, τxy, Vx, Vy, mesh, ipx, ipw, N, dNdX ) 
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = length(ipw)
    ε            = zeros(3)
    τ            = zeros(3)
    m            =  [ 1.0; 1.0; 0.0]
    Dev          =  [ 4/3 -2/3  0.0;
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
        V_ele   = zeros(ndof)
        B       = zeros(ndof, 3)
    
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
            V_ele[1:2:end-1] .= Vx[nodes]
            V_ele[2:2:end]   .= Vy[nodes]
            ε                .= B'*V_ele
            τ                .= ke.*(Dev*ε)
            εkk[e,ip] = ε[1] + ε[2]                                # divergence
            εxx[e,ip], εyy[e,ip], εxy[e,ip] = ε[1], ε[2], ε[3]*0.5 # because of engineering convention
            τxx[e,ip], τyy[e,ip], τxy[e,ip] = τ[1], τ[2], τ[3]
        end
    end
    return nothing
end 

#----------------------------------------------------------#
function ResidualStokesNodalFEM!( Fx, Fy, Fp, Vx, Vy, P, mesh, K_all, Q_all, bV_all )
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
                b_ele      = bV_all[e,:]
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

function ResidualStokesElementalSerialFEM!( Fx, Fy, Fp, Vx, Vy, P, mesh, K_all, Q_all, bV_all )
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
        b_ele      = bV_all[e,:]
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

@views function ElementAssemblyLoopFEM_v2( se, mesh, ipx, ipw, N, dNdX, Vx, Vy, P ) # Adapted from MILAMIN_1.0.1
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = length(ipw)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    K_all_i      = zeros(Int64, mesh.nel, ndof, ndof)
    K_all_j      = zeros(Int64, mesh.nel, ndof, ndof)
    Q_all        = zeros(mesh.nel, ndof, npel)
    Q_all_i      = zeros(Int64, mesh.nel, ndof, npel)
    Q_all_j      = zeros(Int64, mesh.nel, ndof, npel)
    Mi_all       = zeros(mesh.nel, npel, npel)
    bV_all       = zeros(mesh.nel, ndof )
    bV_all_i     = zeros(Int64, mesh.nel, ndof )
    bP_all       = zeros(mesh.nel, npel )
    bP_all_i     = zeros(Int64, mesh.nel, npel )
    K_ele        = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, npel)
    Q_ele_ip     = zeros(ndof, npel)
    M_ele        = zeros(npel ,npel)
    M_inv        = zeros(npel ,npel)
    P_inv        = zeros(npel ,npel)
    b_ele        = zeros(ndof)
    B            = zeros(ndof, 3)
    Bt           = zeros(3, ndof)
    K_ele_ip     = zeros(ndof, ndof)
    Bv           = zeros(ndof)
    x            = zeros(mesh.nnel, 2)
    m            =  [ 1.0; 1.0; 0.0]
    Dev          =  [ 4/3 -2/3  0.0;
                     -2/3  4/3  0.0;
                      0.0  0.0  1.0]
    J            = zeros(2,2)
    invJ         = zeros(2,2)
    P            = ones(npel, npel)
    Pb           = ones(npel)
    dNdx         = zeros(nnel, 2)
    Pi           = zeros(npel)
    nodes        = zeros(Int64, nnel)
    bct          = zeros(Int64, ndof)
    bc_dir       = zeros(ndof)
    dNdXi        = zeros(nnel, 2)
    Ni           = zeros(nnel)
    println("Element loop...")
    # Element loop
    @inbounds for e = 1:mesh.nel
        nodes  .= mesh.e2n[e,:]
        K_ele  .= 0.0
        Q_ele  .= 0.0
        M_ele  .= 0.0
        b_ele  .= 0.0
        M_inv  .= 0.0
        # Deal with BC's
        bct             .= 0
        @inbounds for in=1:nnel
            bc_type         = mesh.bcn[nodes[in]]
            x[in,1]           = mesh.xn[nodes[in]]
            x[in,2]      = mesh.yn[nodes[in]]
            if (bc_type==1 || bc_type==3) bct[in]      = 1 end 
            if (bc_type==1 || bc_type==4) bct[in+nnel] = 1 end 
            bc_dir[in]      = Vx[nodes[in]]
            bc_dir[in+nnel] = Vy[nodes[in]]
            # Deal with indices
            K_all_i[e,in,:]      .= nodes[in]
            K_all_i[e,in+nnel,:] .= nodes[in] + mesh.nn
            K_all_j[e,:,in]      .= nodes[in]
            K_all_j[e,:,in+nnel] .= nodes[in] + mesh.nn
            bV_all_i[e,in]        = nodes[in]
            bV_all_i[e,in+nnel]   = nodes[in] + mesh.nn
            Q_all_i[e,in,:]      .= nodes[in]
            Q_all_i[e,in+nnel,:] .= nodes[in] + mesh.nn    
        end
        @inbounds for jn=1:npel
            Q_all_j[e,:,jn]      .= mesh.e2p[e,jn]
            bP_all_i[e,jn]        = mesh.e2p[e,jn]
            if jn>1 && npel==3
                P[jn,:] .= x[1:3,jn-1]
            end
        end
        # if npel==3 && nnel!=4 P[2:3,:] .= (x[1:3,:])' end
        ke               = mesh.ke[e]
        # Integration loop
        @inbounds for ip=1:nip
            Ni    .= N[ip,:,:]
            dNdXi .= dNdX[ip,:,:]
            mul!(J, x', dNdXi)
            detJ           = J[1,1]*J[2,2] - J[1,2]*J[2,1]
            w              = ipw[ip] * detJ
            invJ[1,1]      = +J[2,2] / detJ
            invJ[1,2]      = -J[1,2] / detJ
            invJ[2,1]      = -J[2,1] / detJ
            invJ[2,2]      = +J[1,1] / detJ
            mul!(dNdx, dNdXi, invJ)                  
            B[1:nnel,1]     .= dNdx[:,1]
            B[nnel+1:end,2] .= dNdx[:,2]
            B[1:nnel,3]     .= dNdx[:,2]
            B[nnel+1:end,3] .= dNdx[:,1]
            Bv[1:nnel]      .= dNdx[:,1]
            Bv[nnel+1:end]  .= dNdx[:,2]
            mul!(Bt, Dev,B')
            mul!(K_ele_ip, B, Bt)
            K_ele        .+= w .* ke .* K_ele_ip
            if npel==3
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
                mul!(Pb[2:3], x', Ni )
                Pi        .= P\Pb
                Q_ele    .-= w .* (Bv*Pi') 
                # M_ele   .+= ipw[ip] .* detJ .* Pi*Pi' # mass matrix P, not needed for incompressible
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
            elseif npel==1  
                mul!(Q_ele_ip, B, m)
                Q_ele    .-= w .* Q_ele_ip          # B*m*Np'
            end
            b_ele[1:2:end]    .+= w .* se[e,1] .* Ni 
            b_ele[nnel+1:end] .+= w .* se[e,2] .* Ni 
        end
        # Element matrices
        @inbounds for jn=1:ndof
            for in=1:ndof
                if bct[jn]==0 && bct[in]==0
                    K_all[e,jn,in] = K_ele[jn,in]
                end
                if bct[jn]==0 && bct[in]==1
                    bV_all[e,jn] -= K_ele[jn,in]*bc_dir[in]
                end
            end
            if bct[jn]==0
                bV_all[e,jn]  += b_ele[jn]
            elseif bct[jn]==1
                K_all[e,jn,jn] = 1.0
                bV_all[e,jn]  = bc_dir[jn]
            end
        end
        @inbounds for jn=1:ndof
            for in=1:npel
                if bct[jn]==0
                    Q_all[e,jn,in] = Q_ele[jn,in]
                else bct[jn]==1
                    bP_all[e,in] += Q_ele[jn,in] * bc_dir[jn]
                end
            end
        end
    end
    println("Sparsification...")
    @time Kuu, Kup, fu, fp = SparsifyStokes( mesh.nn, mesh.np, K_all_i, K_all_j, K_all, Q_all_i, Q_all_j, Q_all, bV_all_i, bV_all, bP_all_i, bP_all )
    return Kuu, Kup, fu, fp
    # return 0, 0, 0, 0
end 
    
function SparsifyStokes( nn, np, K_all_i, K_all_j, K_all, Q_all_i, Q_all_j, Q_all, bV_all_i, bV_all, bP_all_i, bP_all )
    _oneV = ones(size(bV_all_i[:]))
    _oneP = ones(size(bP_all_i[:]))
    Kuu   =       dropzeros(sparse(K_all_i[:], K_all_j[:],     K_all[:], nn*2, nn*2))
    Kup   =       dropzeros(sparse(Q_all_i[:], Q_all_j[:],     Q_all[:], nn*2, np  ))
    fu    = Array(dropzeros(sparse(bV_all_i[:],     _oneV,    bV_all[:], nn*2,  1  )))
    fp    = Array(dropzeros(sparse(bP_all_i[:],     _oneP,    bP_all[:], np,    1  )))
    return Kuu, Kup, fu, fp
end

#----------------------------------------------------------#

function ElementAssemblyLoopFEM_v0( se, mesh, ipx, ipw, N, dNdX ) # Adapted from MILAMIN_1.0.1
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = length(ipw)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    Q_all        = zeros(mesh.nel, ndof, npel)
    Mi_all       = zeros(mesh.nel, npel, npel)
    bV_all        = zeros(mesh.nel, ndof )
    K_ele        =  zeros(ndof, ndof)
    Q_ele        =  zeros(ndof, npel)
    M_ele        =  zeros(npel ,npel)
    M_inv        =  zeros(npel ,npel)
    b_ele        =  zeros(ndof)
    B            =  zeros(ndof, 3)
    m            =  [ 1.0; 1.0; 0.0]
    Dev          =  [ 4/3 -2/3  0.0;
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
                Np       = N[ip,:,1]
                Pb[2:3] .= x'*Np 
                Pi       = P\Pb
                Q_ele   .-= ipw[ip] .* detJ .* (Bvol[:]*Pi') 
                # M_ele   .+= ipw[ip] .* detJ .* Pi*Pi' # mass matrix P, not needed for incompressible
            else
                if npel==1 Np = 1.0        end
                if npel==3 Np = Np0[ip,:,:] end
                Q_ele   .-= ipw[ip] .* detJ .* (B*m*Np') 
            end
            b_ele[1:2:end] .+= ipw[ip] .* detJ .* se[e,1] .* N[ip,:] 
            b_ele[2:2:end] .+= ipw[ip] .* detJ .* se[e,2] .* N[ip,:]
        end
        K_all[e,:,:]  .= K_ele
        Q_all[e,:,:]  .= Q_ele
        bV_all[e,:]    .= b_ele
    end
    return K_all, Q_all, Mi_all, bV_all
end 

#----------------------------------------------------------#


@views function ElementAssemblyLoopFEM_v3( se, mesh, ipx, ipw, N, dNdX, Vx, Vy, P ) # Adapted from MILAMIN_1.0.1
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = length(ipw)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    K_all_i      = zeros(Int64, mesh.nel, ndof, ndof)
    K_all_j      = zeros(Int64, mesh.nel, ndof, ndof)
    Q_all        = zeros(mesh.nel, ndof, npel)
    Q_all_i      = zeros(Int64, mesh.nel, ndof, npel)
    Q_all_j      = zeros(Int64, mesh.nel, ndof, npel)
    Mi_all       = zeros(mesh.nel, npel, npel)
    bV_all       = zeros(mesh.nel, ndof )
    bV_all_i     = zeros(Int64, mesh.nel, ndof )
    bP_all       = zeros(mesh.nel, npel )
    bP_all_i     = zeros(Int64, mesh.nel, npel )
    K_ele        = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, npel)
    Q_ele_ip     = zeros(ndof, npel)
    M_ele        = zeros(npel ,npel)
    M_inv        = zeros(npel ,npel)
    P_inv        = zeros(npel ,npel)
    b_ele        = zeros(ndof)
    B            = zeros(ndof, 3)
    Bt           = zeros(3, ndof)
    K_ele_ip     = zeros(ndof, ndof)
    Bv           = zeros(ndof)
    m            =  [ 1.0; 1.0; 0.0]
    Dev          =  [ 4/3 -2/3  0.0;
                     -2/3  4/3  0.0;
                      0.0  0.0  1.0]
    J            = zeros(2,2)
    invJ         = zeros(2,2)
    P            = @SMatrix ones(npel, npel)
    Pb           = ones(npel)
    dNdx         = zeros(nnel, 2)
    Pi           = zeros(npel)
    nodes        = zeros(Int64, nnel)
    bct          = @SVector zeros(Int64, ndof)
    bc_dir       = @SVector zeros(ndof)
    dNdXi        = @SMatrix zeros(mesh.nnel, 2)
    x            = @SMatrix zeros(mesh.nnel, 2)
    Ni           = @SVector zeros(nnel)
    println("Element loop...")
    # Element loop
    @inbounds for e = 1:mesh.nel
        nodes  .= mesh.e2n[e,:]
        K_ele  .= 0.0
        Q_ele  .= 0.0
        M_ele  .= 0.0
        b_ele  .= 0.0
        M_inv  .= 0.0
        # Deal with BC's
        for in =1:ndof
            @set! bct[in]             = 0
        end
        @inbounds for in=1:nnel
            bc_type       = mesh.bcn[nodes[in]]
            @set! x[in,1] = @views mesh.xn[nodes[in]]
            @set! x[in,2] = @views mesh.yn[nodes[in]]
            @set! dNdXi[in,1] = 1.0
            if (bc_type==1 || bc_type==3) @set! bct[in]      = 1 end 
            if (bc_type==1 || bc_type==4) @set! bct[in+nnel] = 1 end 
            @set! bc_dir[in]      = Vx[nodes[in]]
            @set! bc_dir[in+nnel] = Vy[nodes[in]]
            # Deal with indices
            K_all_i[e,in,:]      .= nodes[in]
            K_all_i[e,in+nnel,:] .= nodes[in] + mesh.nn
            K_all_j[e,:,in]      .= nodes[in]
            K_all_j[e,:,in+nnel] .= nodes[in] + mesh.nn
            bV_all_i[e,in]        = nodes[in]
            bV_all_i[e,in+nnel]   = nodes[in] + mesh.nn
            Q_all_i[e,in,:]      .= nodes[in]
            Q_all_i[e,in+nnel,:] .= nodes[in] + mesh.nn    
        end
        @inbounds for jn=1:npel
            Q_all_j[e,:,jn]      .= mesh.e2p[e,jn]
            bP_all_i[e,jn]        = mesh.e2p[e,jn]
            if jn>1 && npel==3
                @set! P[jn,:] .= x[1:3,jn-1]
            end
        end
        # if npel==3 && nnel!=4 P[2:3,:] .= (x[1:3,:])' end
        ke               = mesh.ke[e]
        # Integration loop
        @inbounds for ip=1:nip
            psize(dNdXi)
            psize(dNdX[ip,:,:])
            # for in=1:1#nnel
            #     # @set! Ni[in] = N[ip,in,1]
            #     @set! dNdXi[in,1] = dNdX[ip,in,1]
            #     # @set! dNdXi[in,2] = @views dNdX[ip,in,2]
            # end

            
            
            mul!(J, x', dNdXi)
            detJ           = J[1,1]*J[2,2] - J[1,2]*J[2,1]
            w              = ipw[ip] * detJ
            invJ[1,1]      = +J[2,2] / detJ
            invJ[1,2]      = -J[1,2] / detJ
            invJ[2,1]      = -J[2,1] / detJ
            invJ[2,2]      = +J[1,1] / detJ
            mul!(dNdx, dNdXi, invJ)                  
            B[1:nnel,1]     .= dNdx[:,1]
            B[nnel+1:end,2] .= dNdx[:,2]
            B[1:nnel,3]     .= dNdx[:,2]
            B[nnel+1:end,3] .= dNdx[:,1]
            Bv[1:nnel]      .= dNdx[:,1]
            Bv[nnel+1:end]  .= dNdx[:,2]
            mul!(Bt, Dev,B')
            mul!(K_ele_ip, B, Bt)
            K_ele        .+= w .* ke .* K_ele_ip
            if npel==3
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
                mul!(Pb[2:3], x', Ni )
                Pi        .= P\Pb
                Q_ele    .-= w .* (Bv*Pi') 
                # M_ele   .+= ipw[ip] .* detJ .* Pi*Pi' # mass matrix P, not needed for incompressible
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
            elseif npel==1  
                mul!(Q_ele_ip, B, m)
                Q_ele    .-= w .* Q_ele_ip          # B*m*Np'
            end
            b_ele[1:2:end]    .+= w .* se[e,1] .* Ni 
            b_ele[nnel+1:end] .+= w .* se[e,2] .* Ni 
        end
        # Element matrices
        @inbounds for jn=1:ndof
            for in=1:ndof
                if bct[jn]==0 && bct[in]==0
                    K_all[e,jn,in] = K_ele[jn,in]
                end
                if bct[jn]==0 && bct[in]==1
                    bV_all[e,jn] -= K_ele[jn,in]*bc_dir[in]
                end
            end
            if bct[jn]==0
                bV_all[e,jn]  += b_ele[jn]
            elseif bct[jn]==1
                K_all[e,jn,jn] = 1.0
                bV_all[e,jn]  = bc_dir[jn]
            end
        end
        @inbounds for jn=1:ndof
            for in=1:npel
                if bct[jn]==0
                    Q_all[e,jn,in] = Q_ele[jn,in]
                else bct[jn]==1
                    bP_all[e,in] += Q_ele[jn,in] * bc_dir[jn]
                end
            end
        end
    end
    println("Sparsification...")
    @time Kuu, Kup, fu, fp = SparsifyStokes( mesh.nn, mesh.np, K_all_i, K_all_j, K_all, Q_all_i, Q_all_j, Q_all, bV_all_i, bV_all, bP_all_i, bP_all )
    return Kuu, Kup, fu, fp
    # return 0, 0, 0, 0
end 