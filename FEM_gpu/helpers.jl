# Helpers for FEM GPU PT script

StabParam(τ, Γ, Ω, mesh_type, ν) = 0.0 # Stabilisation is only needed for FCFV

function ElementAssemblyLoopFEM(mesh, ipw, N, dNdX) # Adapted from MILAMIN_1.0.1
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
    Pr = ones(npel,npel)
    Pb = ones(npel)
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
    return K_all, Q_all#, Mi_all, b
end

# The following functions are borrowed from MILAMIN_1.0.1
function IntegrationTriangle(nip)
    ipx = zeros(nip,2)
    ipw = zeros(nip,1)
    if nip == 1
        # Location
        ipx[1,1] = 1/3
        ipx[1,2] = 1/3
        # Weight
        ipw[1]   = 0.5
    elseif nip == 3
        # Location
        ipx[1,1] = 1/6
        ipx[1,2] = 1/6
        ipx[2,1] = 2/3
        ipx[2,2] = 1/6
        ipx[3,1] = 1/6
        ipx[3,2] = 2/3
        # Weight    
        ipw[1] = 1/6
        ipw[2] = 1/6
        ipw[3] = 1/6
    elseif nip == 6
        g1 = (8-sqrt(10) + sqrt(38-44*sqrt(2/5)))/18;
        g2 = (8-sqrt(10) - sqrt(38-44*sqrt(2/5)))/18;
        ipx[1,1] = 1-2*g1;                  #0.108103018168070;
        ipx[1,2] = g1;                      #0.445948490915965;
        ipx[2,1] = g1;                      #0.445948490915965;
        ipx[2,2] = 1-2*g1;                  #0.108103018168070;
        ipx[3,1] = g1;                      #0.445948490915965;
        ipx[3,2] = g1;                      #0.445948490915965;
        ipx[4,1] = 1-2*g2;                  #0.816847572980459;
        ipx[4,2] = g2;                      #0.091576213509771;
        ipx[5,1] = g2;                      #0.091576213509771;
        ipx[5,2] = 1-2*g2;                  #0.816847572980459;
        ipx[6,1] = g2;                      #0.091576213509771;
        ipx[6,2] = g2;                      #0.091576213509771;

        w1 = (620+sqrt(213125-53320*sqrt(10)))/3720;
        w2 = (620-sqrt(213125-53320*sqrt(10)))/3720;
        ipw[1] =  w1;                       #0.223381589678011;
        ipw[2] =  w1;                       #0.223381589678011;
        ipw[3] =  w1;                       #0.223381589678011;
        ipw[4] =  w2;                       #0.109951743655322;
        ipw[5] =  w2;                       #0.109951743655322;
        ipw[6] =  w2;                       #0.109951743655322;
        ipw     = 0.5*ipw;
    end
    return ipx, ipw
end

function ShapeFunctions(ipx, nip, nnel)
    N    = zeros(nip,nnel,1)
    dNdx = zeros(nip,nnel,2)
    for i=1:nip
        # Local coordinates of integration points
        η2 = ipx[i,1]
        η3 = ipx[i,2]
        η1 = 1.0-η2-η3
        if nnel==3
            N[i,:,:]    .= [η1 η2 η3]'
            dNdx[i,:,:] .= [-1. 1. 0.;  #w.r.t η2
                            -1. 0. 1.]' #w.r.t η3
        elseif nnel==6
            N[i,:,:]    .= [η1*(2*η1-1) η2*(2*η2-1) η3*(2*η3-1) 4*η2*η3  4*η1*η3    4*η1*η2]'
            dNdx[i,:,:] .= [1-4*η1      -1+4*η2     0.          4*η3    -4*η3       4*η1-4*η2;  #w.r.t η2
                            1-4*η1      0.          -1+4*η3     4*η2     4*η1-4*η3 -4*η2]'      #w.r.t η3
        elseif nnel==7
            N[i,:,:]    .= [η1*(2*η1-1)+3*η1*η2*η3  η2*(2*η2-1)+3*η1*η2*η3  η3*(2*η3-1)+3*η1*η2*η3 4*η2*η3-12*η1*η2*η3     4*η1*η3-12*η1*η2*η3           4*η1*η2-12*η1*η2*η3          27*η1*η2*η3]'
            dNdx[i,:,:] .= [1-4*η1+3*η1*η3-3*η2*η3 -1+4*η2+3*η1*η3-3*η2*η3  3*η1*η3-3*η2*η3        4*η3+12*η2*η3-12*η1*η3 -4*η3+12*η2*η3-12*η1*η3        4*η1-4*η2+12*η2*η3-12*η1*η3 -27*η2*η3+27*η1*η3;  #w.r.t η2
                            1-4*η1+3*η1*η2-3*η2*η3 +3*η1*η2-3*η2*η3        -1+4*η3+3*η1*η2-3*η2*η3 4*η2-12*η1*η2+12*η2*η3  4*η1-4*η3-12*η1*η2+12*η2*η3  -4*η2-12*η1*η2+12*η2*η3       27*η1*η2-27*η2*η3]' #w.r.t η3
        end
    end
    return N, dNdx
end
