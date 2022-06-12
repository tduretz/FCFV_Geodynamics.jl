function IntegrationTriangle( nip )
    ipx = zeros(nip,2)
    ipw = zeros(nip,1)
    if nip == 1
            ipx[1,1] = 1/3
            ipx[1,2] = 1/3
            ipw[1]   = 0.5

    elseif nip == 3
            ipx[1,1] = 1/6
            ipx[1,2] = 1/6
            ipx[2,1] = 2/3
            ipx[2,2] = 1/6
            ipx[3,1] = 1/6
            ipx[3,2] = 2/3

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

        η2 = ipx[i,1]
        η3 = ipx[i,2]
        η1 = 1-η2-η3

        if nnel == 3
            N[i,:,:]    .= [η1 η2 η3]'
            dNdx[i,:,:] .= [-1 1 0;  #w.r.t eta2
                            -1 0 1]' #w.r.t eta3
        end

    end

    return N, dNdx

end