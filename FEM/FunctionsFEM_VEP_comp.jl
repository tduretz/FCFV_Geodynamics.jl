
using SparseArrays, ForwardDiff

psize(a)   = println(size(a))
pminmax(a, name) = @printf("min(%s) = %2.2e --- max(%s) = %2.2e\n", name, minimum(a), name, maximum(a))

#----------------------------------------------------------#
function StabParam(τ, Γ, Ω, mesh_type, ν) 
    return 0. # Stabilisation is only needed for FCFV
end

#----------------------------------------------------------#

function CheckYield( τip, P, ΔP, λ, m, pl )
    τii           = sqrt((m.*τip)'*τip)
    τy            = pl.C*pl.cosϕ  + (P+ΔP)*pl.sinϕ + pl.ηvp*λ
    return τii - τy
end

#----------------------------------------------------------#

function Stress_VEP_comp!( τ, ε , η, m, pl, K, Δt )             
    τ[1:3]           .= 2η.*(ε[1:3])
    τ[4]              = ε[4]                      # no pressure feed back P = P
    τii               = sqrt((m.*τ[1:3])'*τ[1:3]) # don't worry, same as: τii = sqrt(0.5*(τip[1]^2 + τip[2]^2) + τip[3]^2)
    τy                = pl.C*pl.cosϕ + τ[4]*pl.sinϕ             # need to add cosϕ to call it Drucker-Prager but this one follows Stokes2D_simpleVEP
    F                 = τii - τy  
    if (F>0.0) 
        λ             = F/(η + pl.ηvp + K*Δt*pl.sinϕ*pl.sinψ)
        τ[1:3]       .= 2η.*(ε[1:3].-0.5*λ.*τ[1:3]./τii)
        τ[4]         += λ*K*Δt*pl.sinψ
    end
end

#----------------------------------------------------------#

# function Stress_VEP_comp2!( τ, ε, η, m, pl, K, Δt )             
#     τ[1:3]           .= 2η.*(ε[1:3])
#     P                 = -K*Δt*ε[4]                    # no pressure feed back P = P
#     τii               = sqrt((m.*τ[1:3])'*τ[1:3]) # don't worry, same as: τii = sqrt(0.5*(τip[1]^2 + τip[2]^2) + τip[3]^2)
#     τy                = pl.C*pl.cosϕ + P*pl.sinϕ             # need to add cosϕ to call it Drucker-Prager but this one follows Stokes2D_simpleVEP
#     F                 = τii - τy  
#     if (F>0.0) 
#         λ             = F/(η + pl.ηvp + K*Δt*pl.sinϕ*pl.sinψ)
#         τ[1:3]       .= 2η.*(ε[1:3].-0.5*λ.*τ[1:3]./τii)
#         τ[4]          = -K*Δt*( ε[4] - λ*pl.sinψ)
#     end
# end

# #----------------------------------------------------------#

```This one is visco-elasto-viscoplastic and computes tangent matrices```
function ComputeStressFEM_v3!( Djac_all, D_all, pl, ηip, Gip, Kip, Δt, εkk, ε, τ0, τ, V, P0, P, ΔP, mesh, dNdx, weight ) 
    celldims     = (4, 4)
    Cell         = SMatrix{celldims..., Float64, prod(celldims)}
    ndof   = 2*mesh.nnel
    nnel   = mesh.nnel
    npel   = mesh.npel
    nip    = size(weight,2)
    εip    = zeros(3)
    εip_vp = zeros(3)
    τip    = zeros(3)
    Dev    = [ 2/3 -1/3  0.0;   -1/3  2/3  0.0;   0.0  0.0  1/2]
    m      = [0.5; 0.5; 1]
    V_ele  = zeros(ndof)
    B      = zeros(ndof, 3)
    τ0ip   = zeros( 3)
    D      = Cell(zeros(4,4))
    Dj     = Cell(zeros(4,4))
    Fchk   = zeros(mesh.nel, nip) 
    τ1     = zeros(4)
    ε1     = zeros(4)
    # Element loop
     for e = 1:mesh.nel
        nodes   = mesh.e2n[e,:]
        Ptrial  = P[e]
        if npel==3 && nnel!=4 P[2:3,:] .= (x[1:3,:])' end
        # Integration loop
        for ip=1:nip
            ΔP[e,ip]          = 0.0
            η                 = ηip[e,ip]
            G                 = Gip[e,ip]
            K                 = Kip[e,ip]
            τ0ip             .= [τ0.xx[e,ip]; τ0.yy[e,ip]; τ0.xy[e,ip]]
            B[1:2:end,1]     .= dNdx[e,ip,:,1]
            B[2:2:end,2]     .= dNdx[e,ip,:,2]
            B[1:2:end,3]     .= dNdx[e,ip,:,2]
            B[2:2:end,3]     .= dNdx[e,ip,:,1]
            V_ele[1:2:end-1] .= V.x[nodes]
            V_ele[2:2:end]   .= V.y[nodes]
            εip              .= Dev*(B'*V_ele) .+ 0.5*τ0ip./(G.*Δt) # effective deviatoric strain rate
            #################### service minimum!
            # τip              .= 2η.*(εip)
            # D .= 2ηip[e,ip].*I(4)
            #################### service minimum!
            τip              .= 2η.*(εip) # compute trial stress 
            F = CheckYield( τip, P[e], ΔP[e,ip], 0.0, m, pl )
            λ = (F>1e-13)* F / (η + pl.ηvp + K*Δt*pl.sinϕ*pl.sinψ)
            # Stress_VEP!( τip ,εip, η, m, C, P[e], sinϕ, ηvp)
            # τii               = sqrt((m.*τip)'*τip)
            # εii               = sqrt((m.*εip)'*εip)
            # ηip[e,ip]         = τii/2.0/εii 
            # Compute Picard operator
            D = Cell(2ηip[e,ip].*I(4))
            # Compute consistent tangent operator using AD
            S_closed   = (τ,ε) -> Stress_VEP_comp!( τ, ε, η, m, pl, K, Δt )
            ε1        .= [εip; P[e]]
            Dj         = Cell(ForwardDiff.jacobian( S_closed, τ1, ε1 ))
            # if e==1 && ip==1
            #     println(Dj)
            # end

            # S_closed   = (τ,ε) -> Stress_VEP_comp2!( τ, ε, η, m, pl, K, Δt )
            # ε1[1:3]   .= εip
            # εip       .= Dev*(B'*V_ele)
            # div        = εip[1] + εip[2]
            # ε1[4]      = div - P0[e]/(K*Δt)
            # Dj         = Cell(ForwardDiff.jacobian( S_closed, τ1, ε1 ))
            # if λ>1e-10
            #     @show Dj
            # end
            # if e==1 && ip==1
            #     println(Dj)
            # end
            τip       .= τ1[1:3]
            ΔP[e,ip]   = τ1[4] - Ptrial 
            Fchk[e,ip] = CheckYield( τip, P[e], ΔP[e,ip], λ, m, pl )
            # Fill cell arrays
            D_all[e,ip]    = D
            Djac_all[e,ip] = Dj
            ####################
            εip              .= Dev*(B'*V_ele)   # true deviatoric strain rate
            εkk[e,ip]         = εip[1] + εip[2]  # divergence
            ε.xx[e,ip], ε.yy[e,ip], ε.xy[e,ip] = εip[1], εip[2], εip[3]
            τ.xx[e,ip], τ.yy[e,ip], τ.xy[e,ip] = τip[1], τip[2], τip[3]
        end
    end
    pminmax(Fchk, "F ")
    pminmax(ΔP,   "ΔP")
    if maximum(Fchk)> 1e-6 error("Yield function is not satisfied!!!!!") end
    return nothing
end 

#----------------------------------------------------------#

@views function ElementAssemblyLoopFEM_v5( D_all_jac, D_all, η, K, Δt, bc, sp, se, mesh, N, dNdx, w, V, P, τ0 )
    nel          = mesh.nel
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = size(w,2)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    K_all_jac    = zeros(mesh.nel, ndof, ndof)
    Q_all        = zeros(mesh.nel, ndof, npel)
    Q_all_jac    = zeros(mesh.nel, ndof, npel)
    M_all        = zeros(mesh.nel, npel, npel)
    bV_all       = zeros(mesh.nel, ndof )
    bP_all       = zeros(mesh.nel, npel )
    K_ele        = zeros(ndof, ndof)
    K_ele_jac    = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, npel)
    Q_ele_jac    = zeros(ndof, npel)
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
    m_jac        = zeros(3)
    Dev          =  [ 2/3 -1/3  0.0;
                     -1/3  2/3  0.0;
                      0.0  0.0  0.5]
    D            = zeros(4,4)
    D_jac        = zeros(4,4)
    P            = ones(npel, npel)
    Pb           = ones(npel)
    Pi           = zeros(npel)
    Ni           = zeros(nnel)
    bct          = zeros(Int64, ndof)
    bcv          = zeros(ndof)
    println("Element loop...")
    # Element loop
    for e = 1:nel
        K_ele     .= 0.0
        K_ele_jac .= 0.0
        Q_ele     .= 0.0
        Q_ele_jac .= 0.0
        M_ele     .= 0.0
        b_ele     .= 0.0
        M_inv     .= 0.0
        # --- for the linear pressure element
        if npel==3
            for jn=2:npel
                P[jn,:] .= x[1:3,jn-1]
            end
        end
        # Integration loop
        for ip=1:nip
            for i=1:3
                m_jac[i] = field(D_all_jac,i,4)[e,ip] 
                for j=1:3
                    D_jac[j,i] = field(D_all_jac,j,i)[e,ip] 
                    D[j,i]     = field(D_all,    j,i)[e,ip] 
                end
            end
            Ni              .= N[ip,:,:]               
            B[1:nnel,1]     .= dNdx[e,ip,:,1]
            B[nnel+1:end,2] .= dNdx[e,ip,:,2]
            B[1:nnel,3]     .= dNdx[e,ip,:,2]
            B[nnel+1:end,3] .= dNdx[e,ip,:,1]
            Bv[1:nnel]      .= dNdx[e,ip,:,1]
            Bv[nnel+1:end]  .= dNdx[e,ip,:,2]
            # Picard Kuu
            mul!(Bt, D[1:3,1:3]*Dev, B')
            mul!(K_ele_ip, B, Bt)
            K_ele          .+= w[e,ip]  * K_ele_ip
            # Jacobian Kuu
            mul!(Bt, D_jac[1:3,1:3]*Dev, B')
            mul!(K_ele_ip, B, Bt)
            K_ele_jac      .+= w[e,ip]  * K_ele_ip
            if npel==3
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
                mul!(Pb[2:3], x', Ni )
                Pi        .= P\Pb
                Q_ele    .-= w[e,ip] .* (Bv*Pi') 
                # M_ele   .+= ipw[ip] .* detJ .* Pi*Pi' # mass matrix P, not needed for incompressible
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
            elseif npel==1  
                # Picard Kpu (minus divergence transpose)
                mul!(Q_ele_ip, B, m)
                Q_ele     .-= w[e,ip] .* Q_ele_ip  
                # Jacobian Kpu (minus divergence transpose)
                mul!(Q_ele_ip, B, (m.-m_jac))
                Q_ele_jac .-= w[e,ip] .* Q_ele_ip          # B*m*Np'
                # Kpp block
                M_ele     .+= w[e,ip]/(K[e]*Δt)            # compressible block ;)
            end
            # Source term
            b_ele[1:nnel]     .+= w[e,ip] .* se[e,1] .* Ni 
            b_ele[nnel+1:end] .+= w[e,ip] .* se[e,2] .* Ni 
        end
        # if norm(Q_ele_jac.-Q_ele)>1e-10
        #     error("gut")
        # end
        StoreAllElementMatrices!( e, bct, bcv, bc, K_all, K_ele, K_all_jac, K_ele_jac, Q_all, Q_ele, Q_all_jac, Q_ele_jac, M_all, M_ele, bV_all, b_ele, bP_all, ndof, npel )    
    end
    @time begin 
        println("Sparsification...")
        Kuu  = SparsifyMatrix( mesh.nn*2, mesh.nn*2, sp.Ki, sp.Kj, K_all     )
        Kuuj = SparsifyMatrix( mesh.nn*2, mesh.nn*2, sp.Ki, sp.Kj, K_all_jac )
        Kup  = SparsifyMatrix( mesh.nn*2, mesh.np,   sp.Qi, sp.Qj, Q_all     )
        Kupj = SparsifyMatrix( mesh.nn*2, mesh.np,   sp.Qi, sp.Qj, Q_all_jac )
        Kpp  = SparsifyMatrix( mesh.np,   mesh.np,   sp.Mi, sp.Mj, M_all     )
        bu   = SparsifyVector( mesh.nn*2, sp.bVi,    bV_all )
        bp   = SparsifyVector( mesh.np,   sp.bPi,    bP_all )
        Kpu  = -Kup'
    end

    return Kuu, Kuuj, Kup, Kupj, Kpu, Kpp, bu, bp
end


#----------------------------------------------------------#

@views function ResidualStokes_v3( bc, se, mesh, N, dNdx, weight, V, P0, P, ΔP, τ, K, Δt )
    nel   = mesh.nel
    nnel  = mesh.nnel
    npel  = mesh.npel
    nip   = size( weight, 2 )
    Fcont = zeros( nel, npel );
    nodes = zeros(Int64, nnel )
    Vx    = zeros( nnel )
    Vy    = zeros( nnel )
    dx    = zeros( nnel)
    dy    = zeros( nnel)
    inx   = zeros(Bool, nnel)
    iny   = zeros(Bool, nnel)
    Fx    = zeros(mesh.nn)
    Fy    = zeros(mesh.nn)
    # Element loop
    for e = 1:mesh.nel
        nodes  .= mesh.e2n[e,:]
        Vx     .= V.x[nodes]
        Vy     .= V.y[nodes]
        # Integration loop
        for ip=1:nip
            w           = weight[e,ip]
            dx         .= dNdx[e,ip,:,1]
            dy         .= dNdx[e,ip,:,2]
            inx        .= bc.type[e,1,:] .== 0
            iny        .= bc.type[e,2,:] .== 0
            Fcont[e]   -= w *  ( dx'*Vx + dy'*Vy + (P[e]-P0[e])/(K[e,ip]*Δt) ) 
            Fx[nodes] .-= w .* ( dx.*τ.xx[e,ip] .+ dy.*τ.xy[e,ip] .- dx.*(P[e]+ΔP[e,ip]) ) .* inx
            Fy[nodes] .-= w .* ( dy.*τ.yy[e,ip] .+ dx.*τ.xy[e,ip] .- dy.*(P[e]+ΔP[e,ip]) ) .* iny
        end
    end
    fu = [Fx; Fy]
    return fu, Fcont, norm(Fx)/sqrt(length(Fx)), norm(Fy)/sqrt(length(Fy)), norm(Fcont)/sqrt(length(Fcont))
end

#----------------------------------------------------------#

function StoreAllElementMatrices!( e, bct, bcv, bc, K_all, K_ele, K_all_jac, K_ele_jac, Q_all, Q_ele, Q_all_jac, Q_ele_jac, M_all, M_ele, bV_all, b_ele, bP_all, ndof, npel )
    # Element matrices
    bct .= [bc.type[e,1,:]; bc.type[e,2,:]]
    bcv .= [bc.val[e,1,:];  bc.val[e,2,:]]
    for jn=1:ndof
        for in=1:ndof
            if bct[jn]==0 && bct[in]==0
                K_all[e,jn,in] = K_ele[jn,in]
                K_all_jac[e,jn,in] = K_ele_jac[jn,in]
            end
            if bct[jn]==0 && bct[in]==1
                bV_all[e,jn] -= K_ele[jn,in]*bcv[in]
            end
        end
        if bct[jn]==0
            bV_all[e,jn]       += b_ele[jn]
        elseif bct[jn]==1
            K_all[e,jn,jn]     = 1.0
            K_all_jac[e,jn,jn] = 1.0
            bV_all[e,jn]       = bcv[jn]
        end
    end
    for jn=1:ndof
        for in=1:npel
            if bct[jn]==0
                Q_all[e,jn,in]     = Q_ele[jn,in]
                Q_all_jac[e,jn,in] = Q_ele_jac[jn,in]
            else bct[jn]==1
                bP_all[e,in] += Q_ele[jn,in] * bcv[jn]
            end
        end
    end
    for jn=1:npel
        for in=1:npel
            M_all[e,jn,in]    = M_ele[jn,in]
        end
    end
    return nothing
end

#----------------------------------------------------------#
