 # The following functions are borrowed from MILAMIN_1.0.1
function IntegrationTriangle( nip )
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

    N        = zeros(nip,nnel,1)
    dNdx     = zeros(nip,nnel,2)

    for i=1:nip
        # Local coordinates of integration points
        ??2 = ipx[i,1]
        ??3 = ipx[i,2]
        ??1 = 1.0-??2-??3
        if nnel==3
            N[i,:,:]    .= [??1 ??2 ??3]'
            dNdx[i,:,:] .= [-1. 1. 0.;  #w.r.t ??2
                            -1. 0. 1.]' #w.r.t ??3
        elseif nnel==4
            N[i,:,:]    .= [??1+3*??1*??2*??3       ??2+3*??1*??2*??3     ??3+3*??1*??2*??3      27*??1*??2*??3]'
            dNdx[i,:,:] .= [-1+3*??1*??3-3*??2*??3  1+3*??1*??3-3*??2*??3 0+3*??1*??3-3*??2*??3 -27*??2*??3+27*??1*??3;  #w.r.t ??2
                            -1+3*??1*??2-3*??2*??3  0+3*??1*??2-3*??2*??3 1+3*??1*??2-3*??2*??3  27*??1*??2-27*??2*??3]' #w.r.t ??3
        elseif nnel==6
            N[i,:,:]    .= [??1*(2*??1-1) ??2*(2*??2-1) ??3*(2*??3-1) 4*??2*??3  4*??1*??3    4*??1*??2]'
            dNdx[i,:,:] .= [1-4*??1      -1+4*??2     0.          4*??3    -4*??3       4*??1-4*??2;  #w.r.t ??2
                            1-4*??1      0.          -1+4*??3     4*??2     4*??1-4*??3 -4*??2]'      #w.r.t ??3
        elseif nnel==7
            N[i,:,:]    .= [??1*(2*??1-1)+3*??1*??2*??3  ??2*(2*??2-1)+3*??1*??2*??3  ??3*(2*??3-1)+3*??1*??2*??3 4*??2*??3-12*??1*??2*??3     4*??1*??3-12*??1*??2*??3           4*??1*??2-12*??1*??2*??3          27*??1*??2*??3]'
            dNdx[i,:,:] .= [1-4*??1+3*??1*??3-3*??2*??3 -1+4*??2+3*??1*??3-3*??2*??3  3*??1*??3-3*??2*??3        4*??3+12*??2*??3-12*??1*??3 -4*??3+12*??2*??3-12*??1*??3        4*??1-4*??2+12*??2*??3-12*??1*??3 -27*??2*??3+27*??1*??3;  #w.r.t ??2
                            1-4*??1+3*??1*??2-3*??2*??3 +3*??1*??2-3*??2*??3        -1+4*??3+3*??1*??2-3*??2*??3 4*??2-12*??1*??2+12*??2*??3  4*??1-4*??3-12*??1*??2+12*??2*??3  -4*??2-12*??1*??2+12*??2*??3       27*??1*??2-27*??2*??3]' #w.r.t ??3
        end
    end
    return N, dNdx
end

function Integration1D( nip )

    ipx = zeros(nip,1)
    ipw = zeros(nip,1)
    
    if nip == 3
        # # Location
        # ipx[1] = -.775
        # ipx[2] = 0.0
        # ipx[3] = .775
        # # Weight    
        # ipw[1] = 5.0/9.0
        # ipw[2] = 8.0/9.0
        # ipw[3] = 5.0/9.0
        # Location
        ipx[1] = -1.0
        ipx[2] = 0.0
        ipx[3] = 1.0
        # Weight    
        ipw[1] = 1.0/3.0
        ipw[2] = 4.0/3.0
        ipw[3] = 1.0/3.0
    end

    return ipx, ipw
end

function ShapeFunctions1D(ipx, nip, nnel)

    N    = zeros(nip,nnel,1)
    dNdx = zeros(nip,nnel,1)

    for i=1:nip

        ?? = ipx[i]

        if nnel==3
            N[i,:,:]    .= [??/2*(??-1)  1.0-??^2 ??/2*(??+1)]'
            dNdx[i,:,:] .= [??-1.0/2.0 -2.0*??   ??+1.0/2.0]'  #w.r.t ??2
        end
    end
    return N, dNdx
end