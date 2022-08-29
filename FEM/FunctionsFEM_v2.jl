
using SparseArrays

psize(a) = println(size(a))

#----------------------------------------------------------#
function StabParam(τ, Γ, Ω, mesh_type, ν) 
    return 0. # Stabilisation is only needed for FCFV
end

#----------------------------------------------------------#

function ComputeStressFEM_v2!( ηip, Gip, Δt, εkk, ε, τ0, τ, V, mesh, dNdx, weight ) 
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = size(weight,2)
    εip          = zeros(3)
    τip          = zeros(3)
    Dev          =  [ 4/3 -2/3  0.0;
                     -2/3  4/3  0.0;
                      0.0  0.0  1.0]
    m            = [0.5; 0.5; 1]
    P      = ones(npel,npel)
    V_ele  = zeros(ndof)
    B      = zeros(ndof, 3)
    τ0ip   = zeros( 3)
    # Element loop
     for e = 1:mesh.nel
        nodes   = mesh.e2n[e,:]
        if npel==3 && nnel!=4 P[2:3,:] .= (x[1:3,:])' end
        # Integration loop
        for ip=1:nip
            η       = ηip[e,ip]
            G       = Gip[e,ip]
            τ0ip   .= [τ0.xx[e,ip]; τ0.yy[e,ip]; τ0.xy[e,ip]]
            B[1:2:end,1]     .= dNdx[e,ip,:,1]
            B[2:2:end,2]     .= dNdx[e,ip,:,2]
            B[1:2:end,3]     .= dNdx[e,ip,:,2]
            B[2:2:end,3]     .= dNdx[e,ip,:,1]
            V_ele[1:2:end-1] .= V.x[nodes]
            V_ele[2:2:end]   .= V.y[nodes]
            εip              .= B'*V_ele .+ m.*τ0ip./(G*Δt) 
            τip              .= η.*(Dev*εip)
            εkk[e,ip]         = εip[1] + εip[2]                        # divergence
            ε.xx[e,ip], ε.yy[e,ip], ε.xy[e,ip] = εip[1], εip[2], εip[3]*0.5 # because of engineering convention
            τ.xx[e,ip], τ.yy[e,ip], τ.xy[e,ip] = τip[1], τip[2], τip[3]
        end
    end
    return nothing
end 

#----------------------------------------------------------#

function ComputeStressFEM_v2!( ηip, εkk, ε, τ, V, mesh, dNdx, weight ) 
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = size(weight,2)
    εip          = zeros(3)
    τip          = zeros(3)
    Dev          =  [ 4/3 -2/3  0.0;
                     -2/3  4/3  0.0;
                      0.0  0.0  1.0]
    P      = ones(npel,npel)
    V_ele  = zeros(ndof)
    B      = zeros(ndof, 3)
    # Element loop
     for e = 1:mesh.nel
        nodes   = mesh.e2n[e,:]
        if npel==3 && nnel!=4 P[2:3,:] .= (x[1:3,:])' end
        # Integration loop
        for ip=1:nip
            ke      = ηip[e,ip]
            B[1:2:end,1]     .= dNdx[e,ip,:,1]
            B[2:2:end,2]     .= dNdx[e,ip,:,2]
            B[1:2:end,3]     .= dNdx[e,ip,:,2]
            B[2:2:end,3]     .= dNdx[e,ip,:,1]
            V_ele[1:2:end-1] .= V.x[nodes]
            V_ele[2:2:end]   .= V.y[nodes]
            εip              .= B'*V_ele
            τip              .= ke.*(Dev*εip)
            εkk[e,ip]         = εip[1] + εip[2]                        # divergence
            ε.xx[e,ip], ε.yy[e,ip], ε.xy[e,ip] = εip[1], εip[2], εip[3]*0.5 # because of engineering convention
            τ.xx[e,ip], τ.yy[e,ip], τ.xy[e,ip] = τip[1], τip[2], τip[3]
        end
    end
    return nothing
end 

#----------------------------------------------------------#

function SparsifyMatrix( nl, nc, Ki, Kj, Kv )
    K    =  dropzeros(sparse(Ki[:], Kj[:], Kv[:], nl, nc))
    return K
end

function SparsifyVector( n, Ki, Kv )
    _one = ones(size(Ki[:]))
    f    = Array(dropzeros(sparse(Ki[:], _one, Kv[:], n, 1 )))
    return f
end
#----------------------------------------------------------#

@views function Eval_dNdx_W_Sparsity( mesh, ipw, N, dNdX, V )
    nel   = mesh.nel
    ndof  = 2*mesh.nnel
    nnel  = mesh.nnel
    npel  = mesh.npel
    nip   = length(ipw)
    K_i   = zeros(Int64, nel, ndof, ndof)
    K_j   = zeros(Int64, nel, ndof, ndof)
    Q_i   = zeros(Int64, nel, ndof, npel)
    Q_j   = zeros(Int64, nel, ndof, npel)
    bV_i  = zeros(Int64, nel, ndof )
    bP_i  = zeros(Int64, nel, npel )
    J     = zeros(2, 2)
    invJ  = zeros(2, 2)
    dNdx  = zeros(nel, nip, nnel, 2)
    w     = zeros(nel, nip)
    nodes = zeros(Int64, nnel)
    bct   = zeros(Int64, nel, 2, nnel)
    bcv   = zeros(nel, 2, nnel)
    dNdXi = zeros(nnel, 2)
    Ni    = zeros(nnel)
    x     = zeros(mesh.nnel, 2)
    println("Element loop...")
    # Element loop
     for e = 1:mesh.nel
        nodes  .= mesh.e2n[e,:]
        # Deal with BC's
        bct[e,:,:] .= 0
         for in=1:nnel
            bc_type       = mesh.bcn[nodes[in]]
            bcv[e,1,in]   = V.x[nodes[in]]
            bcv[e,2,in]   = V.y[nodes[in]] 
            if (bc_type==1 || bc_type==3) bct[e,1,in] = 1 end # 1: full Dirichlet - 3: fixed Vx (free Vy)
            if (bc_type==1 || bc_type==4) bct[e,2,in] = 1 end # 1: full Dirichlet - 4: fixed Vy (free Vx)
            # Element nodal coordinates           
            x[in,1]           = mesh.xn[nodes[in]]
            x[in,2]           = mesh.yn[nodes[in]]
            # Deal with indices
            K_i[e,in,:]      .= nodes[in]
            K_i[e,in+nnel,:] .= nodes[in] + mesh.nn
            K_j[e,:,in]      .= nodes[in]
            K_j[e,:,in+nnel] .= nodes[in] + mesh.nn
            bV_i[e,in]        = nodes[in]
            bV_i[e,in+nnel]   = nodes[in] + mesh.nn
            Q_i[e,in,:]      .= nodes[in]
            Q_i[e,in+nnel,:] .= nodes[in] + mesh.nn    
        end
        for jn=1:npel
            Q_j[e,:,jn]      .= mesh.e2p[e,jn]
            bP_i[e,jn]        = mesh.e2p[e,jn]
        end
        # Integration loop
        for ip=1:nip
            Ni       .= N[ip,:,:]
            dNdXi    .= dNdX[ip,:,:]
            mul!(J, x', dNdXi)
            detJ      = J[1,1]*J[2,2] - J[1,2]*J[2,1]
            w[e,ip]   = ipw[ip] * detJ
            invJ[1,1] = +J[2,2] / detJ
            invJ[1,2] = -J[1,2] / detJ
            invJ[2,1] = -J[2,1] / detJ
            invJ[2,2] = +J[1,1] / detJ
            mul!(dNdx[e,ip,:,:], dNdXi, invJ)                  
        end
    end
    return dNdx, w, (Ki=K_i, Kj=K_j, Qi=Q_i, Qj=Q_j, bVi=bV_i, bPi=bP_i), (type=bct, val=bcv)
end

#----------------------------------------------------------#

@views function ElementAssemblyLoopFEM_v4( bc, sp, se, mesh, N, dNdx, w, V, P, τ0 )
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = size(w,2)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    Q_all        = zeros(mesh.nel, ndof, npel)
    bV_all       = zeros(mesh.nel, ndof )
    bP_all       = zeros(mesh.nel, npel )
    K_ele        = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, npel)
    Q_ele_ip     = zeros(ndof, npel)
    M_ele        = zeros(npel ,npel)
    M_inv        = zeros(npel ,npel)
    P_inv        = zeros(npel ,npel)
    b_ele        = zeros(ndof)
    B            = zeros(ndof, 3)
    Bt           = zeros(3, ndof)
    τ0           = zeros(3)
    K_ele_ip     = zeros(ndof, ndof)
    Bv           = zeros(ndof)
    x            = zeros(mesh.nnel, 2)
    m            =  [ 1.0; 1.0; 0.0]
    Dev          =  [ 4/3 -2/3  0.0;
                     -2/3  4/3  0.0;
                      0.0  0.0  1.0]
    P            = ones(npel, npel)
    Pb           = ones(npel)
    Pi           = zeros(npel)
    Ni           = zeros(nnel)
    bct          = zeros(Int64, ndof)
    bcv          = zeros(ndof)
    println("Element loop...")
    # Element loop
    for e = 1:mesh.nel
        K_ele  .= 0.0
        Q_ele  .= 0.0
        M_ele  .= 0.0
        b_ele  .= 0.0
        M_inv  .= 0.0
        # --- for the linear pressure element
        if npel==3
            for jn=2:npel
                P[jn,:] .= x[1:3,jn-1]
            end
        end
        ke = mesh.ke[e]
        # Integration loop
        for ip=1:nip
            # τ0[1]  = τxx[e,ip]
            # τ0[2]  = τyy[e,ip]
            # τ0[3]  = τxy[e,ip]   
            Ni    .= N[ip,:,:]               
            B[1:nnel,1]     .= dNdx[e,ip,:,1]
            B[nnel+1:end,2] .= dNdx[e,ip,:,2]
            B[1:nnel,3]     .= dNdx[e,ip,:,2]
            B[nnel+1:end,3] .= dNdx[e,ip,:,1]
            Bv[1:nnel]      .= dNdx[e,ip,:,1]
            Bv[nnel+1:end]  .= dNdx[e,ip,:,2]
            mul!(Bt, Dev, B')
            mul!(K_ele_ip, B, Bt)
            K_ele        .+= w[e,ip] .* ke .* K_ele_ip
            if npel==3
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
                mul!(Pb[2:3], x', Ni )
                Pi        .= P\Pb
                Q_ele    .-= w[e,ip] .* (Bv*Pi') 
                # M_ele   .+= ipw[ip] .* detJ .* Pi*Pi' # mass matrix P, not needed for incompressible
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
            elseif npel==1  
                mul!(Q_ele_ip, B, m)
                Q_ele    .-= w[e,ip] .* Q_ele_ip          # B*m*Np'
            end
            # Source term
            b_ele[1:nnel]     .+= w[e,ip] .* se[e,1] .* Ni 
            b_ele[nnel+1:end] .+= w[e,ip] .* se[e,2] .* Ni 
            # Visco-elasticity
            # if e==1
            #     println(τxx[e,ip])
            # end
            # mul!(Bt, Dev, B')
            # if e==1
            #     println(w .* B * τ0)
            # end
            # b_ele .+= w[e,ip] .* B * τ0 
            # b_ele[1:nnel]     .+= w .* dNdx[:,1] * τxx[e,ip] 
            # b_ele[nnel+1:end] .+= w .* dNdx[:,2] * τyy[e,ip] 
        end
        # Element matrices
        bct .= [bc.type[e,1,:]; bc.type[e,2,:]]
        bcv .= [bc.val[e,1,:];  bc.val[e,2,:]]
        for jn=1:ndof
            for in=1:ndof
                if bct[jn]==0 && bct[in]==0
                    K_all[e,jn,in] = K_ele[jn,in]
                end
                if bct[jn]==0 && bct[in]==1
                    bV_all[e,jn] -= K_ele[jn,in]*bcv[in]
                end
            end
            if bct[jn]==0
                bV_all[e,jn]  += b_ele[jn]
            elseif bct[jn]==1
                K_all[e,jn,jn] = 1.0
                bV_all[e,jn]  = bcv[jn]
            end
        end
        for jn=1:ndof
            for in=1:npel
                if bct[jn]==0
                    Q_all[e,jn,in] = Q_ele[jn,in]
                else bct[jn]==1
                    bP_all[e,in] += Q_ele[jn,in] * bcv[jn]
                end
            end
        end
    end
    println("Sparsification...")
    @time begin 
        Kuu = SparsifyMatrix( mesh.nn*2, mesh.nn*2, sp.Ki, sp.Kj, K_all )
        Kup = SparsifyMatrix( mesh.nn*2, mesh.np, sp.Qi, sp.Qj, Q_all )
        fu  = SparsifyVector( mesh.nn*2, sp.bVi, bV_all )
        fp  = SparsifyVector( mesh.np, sp.bPi, bP_all )
    end
    return Kuu, Kup, fu, fp
end

#----------------------------------------------------------#

@views function ResidualStokes_v1( bc, sp, se, mesh, N, dNdx, weight, V, P, τ )
    nel    = mesh.nel
    ndof   = 2*mesh.nnel
    nnel   = mesh.nnel
    npel   = mesh.npel
    nip    = size(weight,2)
    Fmom   = zeros( 2, nel, nnel);
    Fcont  = zeros( nel, npel);
    nodes  = zeros(Int64, nnel)
    Vx     = zeros( nnel)
    Vy     = zeros( nnel)
    # Element loop
    for e = 1:mesh.nel
        nodes  .= mesh.e2n[e,:]
        Vx     .= V.x[nodes]
        Vy     .= V.y[nodes]
        # Integration loop
        for ip=1:nip
            w  = weight[e,ip]
            # Nodal loop
            for in=1:nnel
                dx,  dy  = dNdx[e,ip,in,1],    dNdx[e,ip,in,2]
                inx, iny = bc.type[e,1,in]==0, bc.type[e,1,in]==0
                Fmom[1,e,in] -= w * ( dx*τ.xx[e,ip] + dy*τ.xy[e,ip] - dx*P[e] ) * inx
                Fmom[2,e,in] -= w * ( dy*τ.yy[e,ip] + dx*τ.xy[e,ip] - dy*P[e] ) * iny
                Fcont[e]     -= w * ( dx*Vx[in] + dy*Vy[in] )
            end
        end
    end
    @time F = SparsifyVector( mesh.nn*2, sp.bVi, hcat(Fmom[1,:,:], Fmom[2,:,:]) )
    # @time Fy = SparsifyVector( mesh.nn*2 sp.bVi, Fmom[1,:,:] )
    Fx = F[1:mesh.nn]
    Fy = F[mesh.nn+1:end]
    @printf("||Fx|| = %2.2e\n", norm(Fx)/sqrt(length(Fx)))
    @printf("||Fy|| = %2.2e\n", norm(Fy)/sqrt(length(Fy)))
    @printf("||Fp|| = %2.2e\n", norm(Fcont)/sqrt(length(Fcont)))
    fu = [Fx; Fy]
    return fu, Fcont
end

#----------------------------------------------------------#