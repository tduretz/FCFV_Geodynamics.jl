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
            println(size(N))
            N[i,:,:]    .= [η1*(2*η1-1)+3*η1*η2*η3  η2*(2*η2-1)+3*η1*η2*η3  η3*(2*η3-1)+3*η1*η2*η3 4*η2*η3-12*η1*η2*η3     4*η1*η3-12*η1*η2*η3           4*η1*η2-12*η1*η2*η3          27*η1*η2*η3]'
            dNdx[i,:,:] .= [1-4*η1+3*η1*η3-3*η2*η3 -1+4*η2+3*η1*η3-3*η2*η3  3*η1*η3-3*η2*η3        4*η3+12*η2*η3-12*η1*η3 -4*η3+12*η2*η3-12*η1*η3        4*η1-4*η2+12*η2*η3-12*η1*η3 -27*η2*η3+27*η1*η3;  #w.r.t η2
                            1-4*η1+3*η1*η2-3*η2*η3 +3*η1*η2-3*η2*η3        -1+4*η3+3*η1*η2-3*η2*η3 4*η2-12*η1*η2+12*η2*η3  4*η1-4*η3-12*η1*η2+12*η2*η3  -4*η2-12*η1*η2+12*η2*η3       27*η1*η2-27*η2*η3]' #w.r.t η3
        end
    end

    return N, dNdx

end