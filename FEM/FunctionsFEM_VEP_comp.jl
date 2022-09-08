
using SparseArrays, ForwardDiff

psize(a)   = println(size(a))
pminmax(a) = @printf("min = %2.2e --- max = %2.2e\n", minimum(a), maximum(a))

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

```This one is visco-elasto-viscoplastic and computes tangent matrices```
function ComputeStressFEM_v3!( D_all, pl, ηip, Gip, Kip, Δt, εkk, ε, τ0, τ, V, P, ΔP, mesh, dNdx, weight ) 
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
    D      = zeros(4,4)
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
            # εii               = sqrt((m.*εip)'*εip)
            τip              .= 2η.*(εip) # compute trial stress 
            F = CheckYield( τip, P[e], ΔP[e,ip], 0.0, m, pl )
            λ = (F>1e-13)* F / (η + pl.ηvp + K*Δt*pl.sinϕ*pl.sinψ)
            # Stress_VEP!( τip ,εip, η, m, C, P[e], sinϕ, ηvp)
            # τii               = sqrt((m.*τip)'*τip)
            # ηip[e,ip]         = τii/2.0/εii 
            # D .= 2ηip[e,ip].*I(3)
            # @show D

            S_closed   = (τ,ε) -> Stress_VEP_comp!( τ, ε, η, m, pl, K, Δt )
            ε1        .= [εip; P[e]]
            D         .= ForwardDiff.jacobian( S_closed, τ1, ε1 )
            τip       .= τ1[1:3]
            ΔP[e,ip]   = τ1[4] - Ptrial 
            Fchk[e,ip] = CheckYield( τip, P[e], ΔP[e,ip], λ, m, pl )
            
            for j=1:4
                for i=1:4
                    field(D_all,j,i)[e,ip] = D[j,i]
                end
            end
            ####################
            εip              .= Dev*(B'*V_ele) # true deviatoric strain rate
            εkk[e,ip]         = εip[1] + εip[2]                        # divergence
            ε.xx[e,ip], ε.yy[e,ip], ε.xy[e,ip] = εip[1], εip[2], εip[3]*0.5 # because of engineering convention
            τ.xx[e,ip], τ.yy[e,ip], τ.xy[e,ip] = τip[1], τip[2], τip[3]
        end
    end
    pminmax(Fchk)
    pminmax(ΔP)
    if maximum(Fchk)> 1e-6 error("Yield function is not satisfied!!!!!") end
    return nothing
end 

#----------------------------------------------------------#

@views function ElementAssemblyLoopFEM_v5( D_all, η, K, Δt, bc, sp, se, mesh, N, dNdx, w, V, P, τ0 )
    nel          = mesh.nel
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = size(w,2)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    Q_all        = zeros(mesh.nel, ndof, npel)
    Q_all_div    = zeros(mesh.nel, ndof, npel)
    M_all        = zeros(mesh.nel, npel, npel)
    bV_all       = zeros(mesh.nel, ndof )
    bP_all       = zeros(mesh.nel, npel )
    K_ele        = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, npel)
    Q_ele_div    = zeros(ndof, npel)
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
    mjac         = zeros(3)
    Dev          =  [ 2/3 -1/3  0.0;
                     -1/3  2/3  0.0;
                      0.0  0.0  0.5]
    D            = zeros(4,4)
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
        Q_ele     .= 0.0
        Q_ele_div .= 0.0
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
                mjac[i] = field(D_all,i,4)[e,ip] 
                for j=1:3
                    D[j,i] = field(D_all,j,i)[e,ip] 
                end
            end
            Ni    .= N[ip,:,:]               
            B[1:nnel,1]     .= dNdx[e,ip,:,1]
            B[nnel+1:end,2] .= dNdx[e,ip,:,2]
            B[1:nnel,3]     .= dNdx[e,ip,:,2]
            B[nnel+1:end,3] .= dNdx[e,ip,:,1]
            Bv[1:nnel]      .= dNdx[e,ip,:,1]
            Bv[nnel+1:end]  .= dNdx[e,ip,:,2]
            mul!(Bt, D[1:3,1:3]*Dev, B')
            mul!(K_ele_ip, B, Bt)
            K_ele          .+= w[e,ip]  * K_ele_ip
            if npel==3
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
                mul!(Pb[2:3], x', Ni )
                Pi        .= P\Pb
                Q_ele    .-= w[e,ip] .* (Bv*Pi') 
                # M_ele   .+= ipw[ip] .* detJ .* Pi*Pi' # mass matrix P, not needed for incompressible
                #################### THIS IS NOT OPTMIZED, WILL LIKELY NOT WORK IN 3D ALSO
            elseif npel==1  
                mul!(Q_ele_ip, B, m)
                Q_ele_div.-= w[e,ip] .* Q_ele_ip  
                mul!(Q_ele_ip, B, (m.-mjac))
                Q_ele    .-= w[e,ip] .* Q_ele_ip          # B*m*Np'
                M_ele    .+= w[e,ip]/(K[e]*Δt)            # compressible block ;)

            end
            # Source term
            b_ele[1:nnel]     .+= w[e,ip] .* se[e,1] .* Ni 
            b_ele[nnel+1:end] .+= w[e,ip] .* se[e,2] .* Ni 
        end
        StoreAllElementMatrices!( e, bct, bcv, bc, K_all, K_ele, Q_all, Q_ele, Q_all_div, Q_ele_div, M_all, M_ele, bV_all, b_ele, bP_all, ndof, npel )    
    end
    @time begin 
        println("Sparsification...")
        Kuu = SparsifyMatrix( mesh.nn*2, mesh.nn*2, sp.Ki, sp.Kj, K_all )
        Kupt= SparsifyMatrix( mesh.nn*2, mesh.np, sp.Qi, sp.Qj, Q_all_div )
        Kup = SparsifyMatrix( mesh.nn*2, mesh.np, sp.Qi, sp.Qj, Q_all )
        Kpp = SparsifyMatrix( mesh.np,   mesh.np, sp.Mi, sp.Mj, M_all )
        bu  = SparsifyVector( mesh.nn*2, sp.bVi, bV_all )
        bp  = SparsifyVector( mesh.np, sp.bPi, bP_all )
        Kpu = -Kupt'
    end

    return Kuu, Kup, Kpu, Kpp, bu, bp
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
            Fcont[e]   -= w *  ( dx'*Vx + dy'*Vy + (P[e]-P0[e])/(K[e,ip]*Δt)     ) # linear algebra trick
            Fx[nodes] .-= w .* ( dx.*τ.xx[e,ip] .+ dy.*τ.xy[e,ip] .- dx.*(P[e]+ΔP[e,ip]) ) .* inx
            Fy[nodes] .-= w .* ( dy.*τ.yy[e,ip] .+ dx.*τ.xy[e,ip] .- dy.*(P[e]+ΔP[e,ip]) ) .* iny
        end
    end
    fu = [Fx; Fy]
    return fu, Fcont, norm(Fx)/sqrt(length(Fx)), norm(Fy)/sqrt(length(Fy)), norm(Fcont)/sqrt(length(Fcont))
end

#----------------------------------------------------------#