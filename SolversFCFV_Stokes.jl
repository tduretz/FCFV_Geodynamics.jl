function StokesSolvers(mesh, Kuu, Kup, fu, fp, solver)
    if solver==0
        # Coupled solve
        zero_p = spdiagm(mesh.nel, mesh.nel) 
        K      = [Kuu Kup; Kup' zero_p]
        f      = [fu; fp]
        xh     = K\f
        Vxh    = xh[1:mesh.nf]
        Vyh    = xh[mesh.nf+1:2*mesh.nf]
        Pe     = xh[2*mesh.nf+1:end]  
    elseif solver==1
        # Decoupled solve
        coef  = 1e5*ones(mesh.nel)
        Kppi  = spdiagm(coef)
        Kpu   = -Kup'
        Kuusc = Kuu - Kup*(Kppi*Kpu)
        PC    =  0.5*(Kuusc + Kuusc') 
        t = @elapsed Kf    = cholesky(Symmetric(triu!(PC)))
        @printf("Cholesky took = %02.2e s\n", t)
        u     = zeros(2*mesh.nf,1)
        ru    = zeros(2*mesh.nf, 1)
        fusc  = zeros(2*mesh.nf,1)
        p     = zeros(mesh.nel, 1)
        rp    = zeros(mesh.nel, 1)
        # Ietrations
        for rit=1:10
            ru   .= fu - Kuu*u - Kup*p;
            rp   .= fp - Kpu*u;
            @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, norm(ru)/length(ru), norm(rp)/length(rp))
            if norm(ru)/length(ru) < 1e-10 && norm(rp)/length(rp) < 1e-9
                break
            end
            fusc .=  fu  - Kup*(Kppi*fp + p)
            u    .= Kf\fusc
            p   .+= Kppi*(fp - Kpu*u)
        end
        # Post-process solve
        Vxh = u[1:mesh.nf]
        Vyh = u[mesh.nf+1:2*mesh.nf]
        Pe  = p[:]
    end
    return Vxh, Vyh, Pe
end