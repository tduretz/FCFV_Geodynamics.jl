function ComputeFCFV_o2(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, o2)
    nen   = 3
    ae    = zeros(mesh.nel)
    be    = zeros(mesh.nel,2)
    be_o2 = zeros(mesh.nel,2,nen)
    ze    = zeros(mesh.nel,2,2)
    pe    = zeros(mesh.nel,mesh.nf_el,nen)
    me    = zeros(mesh.nel,nen,nen)
    mei   = zeros(mesh.nel,nen,nen)
    N     = zeros(mesh.nel,mesh.nf_el,nen)
    djx   = zeros(mesh.nel,mesh.nf_el,nen)
    rjx   = zeros(mesh.nel,mesh.nf_el,nen)
    djy   = zeros(mesh.nel,mesh.nf_el,nen)
    rjy   = zeros(mesh.nel,mesh.nf_el,nen)

    # Assemble FCFV elements
    @inbounds for e=1:mesh.nel  

        be[e,1] += mesh.Ω[e]*sex[e]
        be[e,2] += mesh.Ω[e]*sey[e]
        
        for i=1:mesh.nf_el
            
            nodei = mesh.e2f[e,i]
            bc    = mesh.bc[nodei]
            dAi   = mesh.Γ[e,i]
            ni_x  = mesh.n_x[e,i]
            ni_y  = mesh.n_y[e,i]
            taui  = mesh.τ[nodei]                              # Stabilisation parameter for the face

            N[e,i,1] = 1.0
            N[e,i,2] = mesh.xf[nodei] - mesh.xc[e]
            N[e,i,3] = mesh.yf[nodei] - mesh.yc[e]

            pe[e,i,1] = 1.0
            pe[e,i,2] = mesh.xf[nodei] - mesh.xc[e]
            pe[e,i,3] = mesh.yf[nodei] - mesh.yc[e]

            # Assemble
            ze[e,1,1] += (bc==1) * dAi*ni_x*VxDir[nodei] # Dirichlet
            ze[e,1,2] += (bc==1) * dAi*ni_x*VyDir[nodei] # Dirichlet
            ze[e,2,1] += (bc==1) * dAi*ni_y*VxDir[nodei] # Dirichlet
            ze[e,2,2] += (bc==1) * dAi*ni_y*VyDir[nodei] # Dirichlet
            be[e,1]   += (bc==1) * dAi*taui*VxDir[nodei] # Dirichlet
            be[e,2]   += (bc==1) * dAi*taui*VyDir[nodei] # Dirichlet
            ae[e]     +=           dAi*taui
            
        end
    end
    
    if o2==1 # The remaining of this function is only done if second order FCFV is used
        # me matrix
        @inbounds for e=1:mesh.nel 
            for i=1:mesh.nf_el

                nodei = mesh.e2f[e,i]
                bc    = mesh.bc[nodei]
                dAi   = mesh.Γ[e,i] 
                taui  = mesh.τ[nodei]   

                for in=1:nen
                    djx[e,i,in] =  dAi * N[e,i,in] * VxDir[nodei]
                    djy[e,i,in] =  dAi * N[e,i,in] * VyDir[nodei]
                    rjx[e,i,in] =  dAi * N[e,i,in]
                    rjy[e,i,in] =  dAi * N[e,i,in]
                    for jn=1:nen
                        me[e,in,jn] += pe[e,i,jn] * taui * dAi * N[e,i,in]
                    end
                end
            end
        end

        # Inverse of me matrix
        cf11 =  1.0; cf12 =-1.0; cf13 =  1.0;
        cf21 = -1.0; cf22 = 1.0; cf23 = -1.0;
        cf31 =  1.0; cf32 =-1.0; cf33 =  1.0;
        @inbounds for e=1:mesh.nel 
            for i=1:mesh.nf_el
                Mm11 = me[e,2,2] * me[e,3,3] - me[e,2,3] * me[e,3,2]
                Mm12 = me[e,2,1] * me[e,3,3] - me[e,2,3] * me[e,3,1]
                Mm13 = me[e,2,1] * me[e,3,2] - me[e,2,2] * me[e,3,1]
                Mm21 = me[e,1,2] * me[e,3,3] - me[e,1,3] * me[e,3,2]
                Mm22 = me[e,1,1] * me[e,3,3] - me[e,1,3] * me[e,3,1]
                Mm23 = me[e,1,1] * me[e,3,2] - me[e,1,2] * me[e,3,1]
                Mm31 = me[e,1,2] * me[e,2,3] - me[e,1,3] * me[e,2,2]
                Mm32 = me[e,1,1] * me[e,2,3] - me[e,1,3] * me[e,2,1]
                Mm33 = me[e,1,1] * me[e,2,2] - me[e,1,2] * me[e,2,1]
                Ad11 = cf11*Mm11; Ad12 = cf21*Mm21; Ad13 = cf31*Mm31;
                Ad21 = cf12*Mm12; Ad22 = cf22*Mm22; Ad23 = cf32*Mm32;
                Ad31 = cf13*Mm13; Ad32 = cf23*Mm23; Ad33 = cf33*Mm33;
                DetM = me[e,1,1] * Mm11 - me[e,1,2]*Mm12 + me[e,1,3]*Mm13
                mei[e,1,1] = 1.0/DetM*Ad11; mei[e,1,2] = 1.0/DetM*Ad12; mei[e,1,3] = 1.0/DetM*Ad13;
                mei[e,2,1] = 1.0/DetM*Ad21; mei[e,2,2] = 1.0/DetM*Ad22; mei[e,2,3] = 1.0/DetM*Ad23;
                mei[e,3,1] = 1.0/DetM*Ad31; mei[e,3,2] = 1.0/DetM*Ad32; mei[e,3,3] = 1.0/DetM*Ad33;
            end
        end

        # Assembly of b vector
        @inbounds for e=1:mesh.nel

            be_o2[e,1,1] = mesh.Ω[e]*sex[e]
            be_o2[e,2,1] = mesh.Ω[e]*sey[e]

            for i=1:mesh.nf_el
                    
                nodei = mesh.e2f[e,i]
                bc    = mesh.bc[nodei]
                dAi   = mesh.Γ[e,i]
                taui  = mesh.τ[nodei]                              # Stabilisation parameter for the face
                
                for in=1:nen
                    be_o2[e,1,in] += (bc==1) * taui * djx[e,i,in]
                    be_o2[e,2,in] += (bc==1) * taui * djy[e,i,in]
                end
                
            end
        end
    end
    return ae, be, be_o2, ze, pe, mei, pe, rjx, rjy

end

#--------------------------------------------------------------------#

function ComputeElementValues_o2(mesh, Vxh, Vyh, Pe, ae, be, be_o2, ze, rjx, rjy, mei, VxDir, VyDir, o2)
    nen         = 3
    Vxe         = zeros(mesh.nel);
    Vye         = zeros(mesh.nel);
    Vxe1        = zeros(mesh.nel,nen);
    Vye1        = zeros(mesh.nel,nen);
    Vxe2        = zeros(mesh.nel,nen);
    Vye2        = zeros(mesh.nel,nen);
    Txxe        = zeros(mesh.nel);
    Tyye        = zeros(mesh.nel);
    Txye        = zeros(mesh.nel);

    @inbounds for e=1:mesh.nel
    
        Vxe[e]    =  be[e,1]/ae[e]
        Vye[e]    =  be[e,2]/ae[e]
        for in=1:nen
            Vxe1[e,in]  =  be_o2[e,1,in]
            Vye1[e,in]  =  be_o2[e,2,in]
        end
        Txxe[e]   =  mesh.ke[e]/mesh.Ω[e]*ze[e,1,1]
        Tyye[e]   =  mesh.ke[e]/mesh.Ω[e]*ze[e,2,2] 
        Txye[e]   =  mesh.ke[e]/mesh.Ω[e]*0.5*(ze[e,1,2]+ze[e,2,1])
        
        for i=1:mesh.nf_el
            
            # Face
            nodei = mesh.e2f[e,i]
            bc    = mesh.bc[nodei]
            dAi   = mesh.Γ[e,i]
            ni_x  = mesh.n_x[e,i]
            ni_y  = mesh.n_y[e,i]
            taui  = mesh.τ[nodei]  

            # First order
            Vxe[e]  += (bc!=1) *  dAi*taui*Vxh[nodei]/ae[e]
            Vye[e]  += (bc!=1) *  dAi*taui*Vyh[nodei]/ae[e]

            # Second order
            for in=1:nen
                Vxe1[e,in] += (bc!=1) * taui*rjx[e,i,in]*Vxh[nodei]
                Vye1[e,in] += (bc!=1) * taui*rjy[e,i,in]*Vyh[nodei]
            end

            # Assemble
            Txxe[e] += (bc!=1) *  mesh.ke[e]/mesh.Ω[e]*dAi*ni_x*Vxh[nodei]
            Tyye[e] += (bc!=1) *  mesh.ke[e]/mesh.Ω[e]*dAi*ni_y*Vyh[nodei]
            Txye[e] += (bc!=1) *  mesh.ke[e]*0.5*( 1.0/mesh.Ω[e]*dAi*( ni_x*Vyh[nodei] + ni_y*Vxh[nodei] ) )
        end
        Txxe[e] *= 2.0
        Tyye[e] *= 2.0
        Txye[e] *= 2.0
        # inv(me) * ue
        Vxe2[e,1] = (o2==1) * (mei[e,1,1] * Vxe1[e,1] + mei[e,1,2] * Vxe1[e,2] + mei[e,1,3] * Vxe1[e,3])
        Vxe2[e,2] = (o2==1) * (mei[e,2,1] * Vxe1[e,1] + mei[e,2,2] * Vxe1[e,2] + mei[e,2,3] * Vxe1[e,3])
        Vxe2[e,3] = (o2==1) * (mei[e,3,1] * Vxe1[e,1] + mei[e,3,2] * Vxe1[e,2] + mei[e,3,3] * Vxe1[e,3])
        # inv(me) * ue
        Vye2[e,1] = (o2==1) * (mei[e,1,1] * Vye1[e,1] + mei[e,1,2] * Vye1[e,2] + mei[e,1,3] * Vye1[e,3])
        Vye2[e,2] = (o2==1) * (mei[e,2,1] * Vye1[e,1] + mei[e,2,2] * Vye1[e,2] + mei[e,2,3] * Vye1[e,3])
        Vye2[e,3] = (o2==1) * (mei[e,3,1] * Vye1[e,1] + mei[e,3,2] * Vye1[e,2] + mei[e,3,3] * Vye1[e,3])
    end

    if o2==1 Vxe .= Vxe2[:,1] end # extract first values of the ue solution vector (linear function evaluated at cell centroid)
    if o2==1 Vye .= Vye2[:,1] end # extract first values of the ue solution vector (linear function evaluated at cell centroid)

    return Vxe, Vye, Txxe, Tyye, Txye
end

#--------------------------------------------------------------------#

function ElementAssemblyLoop_o2(mesh, ae, be, be_o2, ze, mei, pe, rjx, rjy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, o2, new)
    
    nf  = mesh.nf_el
    σxxNeu =  SxxNeu
    σxyNeu =  SxyNeu
    σyyNeu =  SyyNeu
    σyxNeu =  SyxNeu
    α = ae; β = be; Ζ = ze;

    # Assemble element matrices and rhs
    # Kuuv = zeros(mesh.nel, 2*mesh.nf_el, 2*mesh.nf_el)
    # Muuv = zeros(mesh.nel, 2*mesh.nf_el, 2*mesh.nf_el)
    # Kuui = zeros(mesh.nel, 2*mesh.nf_el, 2*mesh.nf_el)
    # Kuuj = zeros(mesh.nel, 2*mesh.nf_el, 2*mesh.nf_el)
    # fu   = zeros(mesh.nel, 2*mesh.nf_el)
    # Kupi = zeros(mesh.nel, 2*mesh.nf_el)
    # Kupj = zeros(mesh.nel, 2*mesh.nf_el)
    # Kupv = zeros(mesh.nel, 2*mesh.nf_el)
    # fp   = zeros(mesh.nel)

    Kuui = zeros(2*mesh.nf_el, 2*mesh.nf_el, mesh.nel)
    Kuuj = zeros(2*mesh.nf_el, 2*mesh.nf_el, mesh.nel)
    Kuuv = zeros(2*mesh.nf_el, 2*mesh.nf_el, mesh.nel)
    Muuv = zeros(2*mesh.nf_el, 2*mesh.nf_el, mesh.nel)
    fu   = zeros(2*mesh.nf_el, mesh.nel)
    Kupi = zeros(2*mesh.nf_el, mesh.nel)
    Kupj = zeros(2*mesh.nf_el, mesh.nel)
    Kupv = zeros(2*mesh.nf_el, mesh.nel)
    fp   = zeros(mesh.nel)
    nf   = mesh.nf_el

    for e=1:mesh.nel 

        # Element properties
        Ωe = mesh.Ω[e]
        ηe = mesh.ke[e]

        for i=1:mesh.nf_el 

            ni_x, ni_y = mesh.n_x[e,i], mesh.n_y[e,i]
            nodei = mesh.e2f[e,i]
            bci   = mesh.bc[nodei]
            ȷ     = 0.0 + (bci==-1)*1.0# + (bci==0)*1.0 # indicates interface
            # println(ȷ, ' ', new)
            Γi    = mesh.Γ[e,i]
            τi    = mesh.τ[nodei]  

            for j=1:mesh.nf_el

                nj_x, nj_y  = mesh.n_x[e,j], mesh.n_y[e,j]
                nodej = mesh.e2f[e,j]
                bcj   = mesh.bc[nodej]   
                Γj    = mesh.Γ[e,j]
                τj    = mesh.τ[nodej]  
                δ     = 0.0 + (i==j)*1.0    # Delta operator
                on    = (bci!=1) & (bcj!=1) # Activate nodal connection if not Dirichlet!
                        
                # Element matrix components
                ninj = ni_x*nj_x + ni_y*nj_y

                res     = 1.0/ae[e] * Γj 

                # inv(me) * be_x
                meidx1  = mei[e,1,1] * rjx[e,j,1] + mei[e,1,2] * rjx[e,j,2] + mei[e,1,3] * rjx[e,j,3]
                meidx2  = mei[e,2,1] * rjx[e,j,1] + mei[e,2,2] * rjx[e,j,2] + mei[e,2,3] * rjx[e,j,3]
                meidx3  = mei[e,3,1] * rjx[e,j,1] + mei[e,3,2] * rjx[e,j,2] + mei[e,3,3] * rjx[e,j,3]
                rx      = (pe[e,i,1] * meidx1 + pe[e,i,2] * meidx2 + pe[e,i,3] * meidx3)
                resx    = (o2==0) * res + (o2==1) * rx
                # inv(me) * be_y
                meidy1  = mei[e,1,1] * rjy[e,j,1] + mei[e,1,2] * rjy[e,j,2] + mei[e,1,3] * rjy[e,j,3]
                meidy2  = mei[e,2,1] * rjy[e,j,1] + mei[e,2,2] * rjy[e,j,2] + mei[e,2,3] * rjy[e,j,3]
                meidy3  = mei[e,3,1] * rjy[e,j,1] + mei[e,3,2] * rjy[e,j,2] + mei[e,3,3] * rjy[e,j,3]
                ry      = (pe[e,i,1] * meidy1 + pe[e,i,2] * meidy2 + pe[e,i,3] * meidy3)
                resy    = (o2==0) * res + (o2==1) * ry

                # Element matrix 
                Kuuv[j   , i   , e] = on * -Γi * (resx*τi*τj - ηe*Ωe^-1*Γj*(ninj + new*ȷ*ni_x*nj_x) - τi*δ) # u1u1
                Kuuv[j+nf, i   , e] = on * -Γi * (                 - ηe*Ωe^-1*Γj*(       new*ȷ*ni_y*nj_x)       ) # u1u2
                Kuuv[j   , i+nf, e] = on * -Γi * (                 - ηe*Ωe^-1*Γj*(       new*ȷ*ni_x*nj_y)       ) # u2u1
                Kuuv[j+nf, i+nf, e] = on * -Γi * (resy*τi*τj - ηe*Ωe^-1*Γj*(ninj + new*ȷ*ni_y*nj_y) - τi*δ) # u2u2

                # PC - deactivate terms from new interface implementation
                Muuv[j   , i   , e] = on * -Γi * (α[e]^-1*τi*τj*Γj - ηe*Ωe^-1*Γj*(ninj + 0*new*ȷ*ni_x*nj_x) - τi*δ) # u1u1
                Muuv[j+nf, i+nf, e] = on * -Γi * (α[e]^-1*τi*τj*Γj - ηe*Ωe^-1*Γj*(ninj + 0*new*ȷ*ni_y*nj_y) - τi*δ) # u2u2

                # Connectivity
                Kuui[j   , i   , e]  = nodei;         Kuui[j+nf, i   , e]  = nodei
                Kuuj[j   , i   , e]  = nodej;         Kuuj[j+nf, i   , e]  = nodej+mesh.nf
                Kuui[j   , i+nf, e]  = nodei+mesh.nf; Kuui[j+nf, i+nf, e]  = nodei+mesh.nf
                Kuuj[j   , i+nf, e]  = nodej;         Kuuj[j+nf, i+nf, e]  = nodej+mesh.nf
            end 
            # RHS
            Xi    = 0.0 + (bci==2)*1.0
            tix   = ni_x*σxxNeu[nodei] + ni_y*σxyNeu[nodei]
            tiy   = ni_x*σyxNeu[nodei] + ni_y*σyyNeu[nodei]   
            niΖ_x = ni_x*(Ζ[e,1,1] +  new*ȷ*Ζ[e,1,1]) + ni_y*(Ζ[e,2,1] + new*ȷ*Ζ[e,1,2]) 
            niΖ_y = ni_x*(Ζ[e,1,2] +  new*ȷ*Ζ[e,2,1]) + ni_y*(Ζ[e,2,2] + new*ȷ*Ζ[e,2,2])
            #-----------------------------
            ax1    = 1.0/ae[e]*be[e,1] 
            # inv(me) * be_x
            meidx1  = mei[e,1,1] * be_o2[e,1,1] + mei[e,1,2] * be_o2[e,1,2] + mei[e,1,3] * be_o2[e,1,3]
            meidx2  = mei[e,2,1] * be_o2[e,1,1] + mei[e,2,2] * be_o2[e,1,2] + mei[e,2,3] * be_o2[e,1,3]
            meidx3  = mei[e,3,1] * be_o2[e,1,1] + mei[e,3,2] * be_o2[e,1,2] + mei[e,3,3] * be_o2[e,1,3]
            ax2    = (pe[e,i,1] * meidx1 + pe[e,i,2] * meidx2 + pe[e,i,3] * meidx3)
            ax     = (o2==0) * ax1 + (o2==1) * ax2
            #-----------------------------
            ay1    = 1.0/ae[e]*be[e,2] 
            # inv(me) * be_y
            meidy1  = mei[e,1,1] * be_o2[e,2,1] + mei[e,1,2] * be_o2[e,2,2] + mei[e,1,3] * be_o2[e,2,3]
            meidy2  = mei[e,2,1] * be_o2[e,2,1] + mei[e,2,2] * be_o2[e,2,2] + mei[e,2,3] * be_o2[e,2,3]
            meidy3  = mei[e,3,1] * be_o2[e,2,1] + mei[e,3,2] * be_o2[e,2,2] + mei[e,3,3] * be_o2[e,2,3]
            ay2    = (pe[e,i,1] * meidy1 + pe[e,i,2] * meidy2 + pe[e,i,3] * meidy3)
            ay     = (o2==0) * ay1 + (o2==1) * ay2
            #-----------------------------
            feix  = (bci!=1) * -Γi * (-ax*τi + ηe*Ωe^-1*niΖ_x - tix*Xi - (1-new)*ȷ*gbar[e,i,1])
            feiy  = (bci!=1) * -Γi * (-ay*τi + ηe*Ωe^-1*niΖ_y - tiy*Xi - (1-new)*ȷ*gbar[e,i,2])
            # up block
            Kupv[i   , e] -= (bci!=1) * Γi*ni_x
            Kupv[i+nf, e] -= (bci!=1) * Γi*ni_y
            Kupi[i   , e]  = nodei
            Kupj[i   , e]  = e
            Kupi[i+nf, e]  = nodei + mesh.nf
            Kupj[i+nf, e]  = e
            # Dirichlet nodes - uu block
            Kuuv[i   , i   , e] += (bci==1) * 1e0
            Kuuv[i+nf, i+nf, e] += (bci==1) * 1e0
            Muuv[i   , i   , e] += (bci==1) * 1e0
            Muuv[i+nf, i+nf, e] += (bci==1) * 1e0
            fu[i   ,e]          += (bci!=1) * feix + (bci==1) * VxDir[nodei] * 1e0
            fu[i+nf,e]          += (bci!=1) * feiy + (bci==1) * VyDir[nodei] * 1e0
            # Dirichlet nodes - pressure RHS
            fp[e]               -= (bci==1) * Γi*(VxDir[nodei]*ni_x + VyDir[nodei]*ni_y)
        end
    end

    # @inbounds for e=1:mesh.nel 

    #     for i=1:mesh.nf_el 

    #         nodei = mesh.e2f[e,i]
    #         bci   = mesh.bc[nodei]
    #         dAi   = mesh.Γ[e,i]
    #         ni_x  = mesh.n_x[e,i]
    #         ni_y  = mesh.n_y[e,i]
    #         taui  = mesh.τ[nodei]   
                
    #         for j=1:mesh.nf_el

    #             nodej = mesh.e2f[e,j]
    #             bcj   = mesh.bc[nodej]   
    #             dAj   = mesh.Γ[e,j]
    #             nj_x  = mesh.n_x[e,j]
    #             nj_y  = mesh.n_y[e,j]
    #             tauj  = mesh.τ[nodej]   
                        
    #             # Delta
    #             del = 0.0 + (i==j)*1.0
                        
    #             # Element matrix
    #             nitnj   = ni_x*nj_x + ni_y*nj_y;

    #             res     = 1.0/ae[e] * dAj 
                # # inv(me) * be_x
                # meidx1  = mei[e,1,1] * rjx[e,j,1] + mei[e,1,2] * rjx[e,j,2] + mei[e,1,3] * rjx[e,j,3]
                # meidx2  = mei[e,2,1] * rjx[e,j,1] + mei[e,2,2] * rjx[e,j,2] + mei[e,2,3] * rjx[e,j,3]
                # meidx3  = mei[e,3,1] * rjx[e,j,1] + mei[e,3,2] * rjx[e,j,2] + mei[e,3,3] * rjx[e,j,3]
                # rx      = (pe[e,i,1] * meidx1 + pe[e,i,2] * meidx2 + pe[e,i,3] * meidx3)
                # resx    = (o2==0) * res + (o2==1) * rx
                # # inv(me) * be_y
                # meidy1  = mei[e,1,1] * rjy[e,j,1] + mei[e,1,2] * rjy[e,j,2] + mei[e,1,3] * rjy[e,j,3]
                # meidy2  = mei[e,2,1] * rjy[e,j,1] + mei[e,2,2] * rjy[e,j,2] + mei[e,2,3] * rjy[e,j,3]
                # meidy3  = mei[e,3,1] * rjy[e,j,1] + mei[e,3,2] * rjy[e,j,2] + mei[e,3,3] * rjy[e,j,3]
                # ry      = (pe[e,i,1] * meidy1 + pe[e,i,2] * meidy2 + pe[e,i,3] * meidy3)
                # resy    = (o2==0) * res + (o2==1) * ry

    #             Kexij =-dAi * (resx * taui*tauj - mesh.ke[e]/mesh.Ω[e]*dAj*nitnj - taui*del)
    #             Keyij =-dAi * (resy * taui*tauj - mesh.ke[e]/mesh.Ω[e]*dAj*nitnj - taui*del)
    #             yes   = (bci!=1) & (bcj!=1)
    #             Kuuv[e,i,           j           ] = yes * Kexij
    #             Kuuv[e,i+mesh.nf_el,j+mesh.nf_el] = yes * Keyij
    #             Muuv[e,i,           j           ] = yes * Kexij
    #             Muuv[e,i+mesh.nf_el,j+mesh.nf_el] = yes * Keyij
    #             # Connectivity
    #             Kuui[e,i,           j           ]  = nodei;         #Kuui[e,i,           j+mesh.nf_el]  = nodei
    #             Kuuj[e,i,           j           ]  = nodej;         #Kuuj[e,i,           j+mesh.nf_el]  = nodej+mesh.nf
    #             Kuui[e,i+mesh.nf_el,j+mesh.nf_el]  = nodei+mesh.nf; #Kuui[e,i+mesh.nf_el,j           ]  = nodei+mesh.nf; 
    #             Kuuj[e,i+mesh.nf_el,j+mesh.nf_el]  = nodej+mesh.nf; #Kuuj[e,i+mesh.nf_el,j           ]  = nodej;         
    #         end
    #         # RHS
    #         Xi      = 0.0 + (bci==2)*1.0
    #         Ji      = 0.0 + (bci==-1)*1.0
    #         tix     = ni_x*SxxNeu[nodei] + ni_y*SxyNeu[nodei]
    #         tiy     = ni_x*SyxNeu[nodei] + ni_y*SyyNeu[nodei]
    #         nitze_x = ni_x*ze[e,1,1] + ni_y*ze[e,2,1]
    #         nitze_y = ni_x*ze[e,1,2] + ni_y*ze[e,2,2]
            # #-----------------------------
            # ax1    = 1.0/ae[e]*be[e,1] 
            # # inv(me) * be_x
            # meidx1  = mei[e,1,1] * be_o2[e,1,1] + mei[e,1,2] * be_o2[e,1,2] + mei[e,1,3] * be_o2[e,1,3]
            # meidx2  = mei[e,2,1] * be_o2[e,1,1] + mei[e,2,2] * be_o2[e,1,2] + mei[e,2,3] * be_o2[e,1,3]
            # meidx3  = mei[e,3,1] * be_o2[e,1,1] + mei[e,3,2] * be_o2[e,1,2] + mei[e,3,3] * be_o2[e,1,3]
            # ax2    = (pe[e,i,1] * meidx1 + pe[e,i,2] * meidx2 + pe[e,i,3] * meidx3)
            # ax     = (o2==0) * ax1 + (o2==1) * ax2
            # #-----------------------------
            # ay1    = 1.0/ae[e]*be[e,2] 
            # # inv(me) * be_y
            # meidy1  = mei[e,1,1] * be_o2[e,2,1] + mei[e,1,2] * be_o2[e,2,2] + mei[e,1,3] * be_o2[e,2,3]
            # meidy2  = mei[e,2,1] * be_o2[e,2,1] + mei[e,2,2] * be_o2[e,2,2] + mei[e,2,3] * be_o2[e,2,3]
            # meidy3  = mei[e,3,1] * be_o2[e,2,1] + mei[e,3,2] * be_o2[e,2,2] + mei[e,3,3] * be_o2[e,2,3]
            # ay2    = (pe[e,i,1] * meidy1 + pe[e,i,2] * meidy2 + pe[e,i,3] * meidy3)
            # ay     = (o2==0) * ay1 + (o2==1) * ay2
            # #-----------------------------
            # feix    = (bci!=1) * -dAi * (mesh.ke[e]/mesh.Ω[e]*nitze_x - tix*Xi - gbar[e,i,1]*Ji - ax*taui)
            # feiy    = (bci!=1) * -dAi * (mesh.ke[e]/mesh.Ω[e]*nitze_y - tiy*Xi - gbar[e,i,2]*Ji - ay*taui)
    #         # up block
    #         Kupv[e, i]            -= (bci!=1) * dAi*ni_x;
    #         Kupv[e, i+mesh.nf_el] -= (bci!=1) * dAi*ni_y;
    #         Kupi[e, i]             = nodei
    #         Kupj[e, i]             = e
    #         Kupi[e, i+mesh.nf_el]  = nodei + mesh.nf
    #         Kupj[e, i+mesh.nf_el]  = e
    #         # Dirichlet nodes - uu block
    #         Kuuv[e,i,i]                       += (bci==1) * 1e0
    #         Kuuv[e,i+mesh.nf_el,i+mesh.nf_el] += (bci==1) * 1e0
    #         Muuv[e,i,i]                       += (bci==1) * 1e0
    #         Muuv[e,i+mesh.nf_el,i+mesh.nf_el] += (bci==1) * 1e0
    #         fu[e,i]                              += (bci!=1) * feix + (bci==1) * VxDir[nodei] * 1e0
    #         fu[e,i+mesh.nf_el]                   += (bci!=1) * feiy + (bci==1) * VyDir[nodei] * 1e0
    #         # Dirichlet nodes - pressure RHS
    #         # fp[e]                                  = dAi#*ni_x# + VyDir[nodei]*ni_y)

    #         fp[e]                                  -= (bci==1) * -dAi*(VxDir[nodei]*ni_x + VyDir[nodei]*ni_y)# * -1.0# #(bci==1) * -dAi*(VxDir[nodei]*ni_x + VyDir[nodei]*ni_y)
    #     end
    # end
    # Call sparse assembly
    tsparse = @elapsed Kuu, Muu, Kup, fu = Sparsify( Kuui, Kuuj, Kuuv, Muuv, Kupi, Kupj, Kupv, fu, mesh.nf, mesh.nel)

    return Kuu, Muu, Kup, fu, fp, tsparse
end

#--------------------------------------------------------------------#

function Sparsify( Kuui, Kuuj, Kuuv, Muuv, Kupi, Kupj, Kupv, fuv, nf, nel)

    _one = ones(size(Kupi[:]))

    Kuu  =       dropzeros(sparse(Kuui[:], Kuuj[:], Kuuv[:], nf*2, nf*2))
    Muu  =       dropzeros(sparse(Kuui[:], Kuuj[:], Muuv[:], nf*2, nf*2))
    Kup  =       dropzeros(sparse(Kupi[:], Kupj[:], Kupv[:], nf*2, nel ))
    fu   = Array(dropzeros(sparse(Kupi[:],    _one,  fuv[:], nf*2,   1 )))

    # file = matopen(string(@__DIR__,"/results/matrix_K.mat"), "w" )
    # write(file, "Kuu",    Kuu )
    # write(file, "Kup",    Kup )
    # close(file)

    return Kuu, Muu, Kup, fu
end

#--------------------------------------------------------------------#

# function CreateTripletsSparse(mesh, Kuu_v, fu_v, Kup_v)
#     # Create triplets and assemble sparse matrix fo Kuu
#     e2fu = mesh.e2f
#     e2fv = mesh.e2f .+ mesh.nf 
#     e2f  = hcat(e2fu, e2fv)
#     idof = 1:mesh.nf_el*2  
#     ii   = repeat(idof, 1, length(idof))'
#     ij   = repeat(idof, 1, length(idof))
#     Ki   = e2f[:,ii]
#     Kif  = e2f[:,ii[1,:]]
#     Kj   = e2f[:,ij]
#     Kuu  = sparse(Ki[:], Kj[:], Kuu_v[:], mesh.nf*2, mesh.nf*2)
#     # file = matopen(string(@__DIR__,"/results/matrix_uu.mat"), "w" )
#     # write(file, "Ki",       Ki[:] )
#     # write(file, "Kj",    Kj[:] )
#     # write(file, "Kuu",  Kuu_v[:] )
#     # write(file, "nrow",  mesh.nf*2 )
#     # write(file, "ncol",  mesh.nf*2 )
#     # close(file)
#     fu   = sparse(Kif[:], ones(size(Kif[:])), fu_v[:], mesh.nf*2, 1)
#     fu   = Array(fu)
#     droptol!(Kuu, 1e-6)
#     # Create triplets and assemble sparse matrix fo Kup
#     idof = 1:mesh.nf_el*2  
#     ii   = repeat(idof, 1, mesh.nel)'
#     ij   = repeat(1:mesh.nel, 1, length(idof))
#     Ki   = e2f
#     Kj   = ij
#     Kup  = sparse(Ki[:], Kj[:], Kup_v[:], mesh.nf*2, mesh.nel  )
#     # file = matopen(string(@__DIR__,"/results/matrix_up.mat"), "w" )
#     # write(file, "Ki",       Ki[:] )
#     # write(file, "Kj",    Kj[:] )
#     # write(file, "Kup",  Kup_v[:] )
#     # write(file, "nrow",  mesh.nf*2 )
#     # write(file, "ncol",  mesh.nel )
#     # close(file)
#     return Kuu, fu, Kup
# end

#--------------------------------------------------------------------#