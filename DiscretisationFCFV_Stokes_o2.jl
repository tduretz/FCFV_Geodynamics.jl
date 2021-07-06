function ComputeFCFV_o2(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, tau, o2)
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
    @tturbo for iel=1:mesh.nel  

        be[iel,1] += mesh.vole[iel]*sex[iel]
        be[iel,2] += mesh.vole[iel]*sey[iel]
        
        for ifac=1:mesh.nf_el
            
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel],mesh.type,mesh.ke[iel])                              # Stabilisation parameter for the face

            N[iel,ifac,1] = 1.0
            N[iel,ifac,2] = mesh.xf[nodei] - mesh.xc[iel]
            N[iel,ifac,3] = mesh.yf[nodei] - mesh.yc[iel]

            pe[iel,ifac,1] = 1.0
            pe[iel,ifac,2] = mesh.xf[nodei] - mesh.xc[iel]
            pe[iel,ifac,3] = mesh.yf[nodei] - mesh.yc[iel]

            # Assemble
            ze[iel,1,1] += (bc==1) * dAi*ni_x*VxDir[nodei] # Dirichlet
            ze[iel,1,2] += (bc==1) * dAi*ni_x*VyDir[nodei] # Dirichlet
            ze[iel,2,1] += (bc==1) * dAi*ni_y*VxDir[nodei] # Dirichlet
            ze[iel,2,2] += (bc==1) * dAi*ni_y*VyDir[nodei] # Dirichlet
            be[iel,1]   += (bc==1) * dAi*taui*VxDir[nodei] # Dirichlet
            be[iel,2]   += (bc==1) * dAi*taui*VyDir[nodei] # Dirichlet
            ae[iel]     +=           dAi*taui
            
        end
    end
    
    if o2==1 # The remaining of this function is only done if second order FCFV is used
        # me matrix
        @tturbo for iel=1:mesh.nel 
            for ifac=1:mesh.nf_el

                nodei = mesh.e2f[iel,ifac]
                bc    = mesh.bc[nodei]
                dAi   = mesh.dA[iel,ifac] 
                taui  = StabParam(tau,dAi,mesh.vole[iel],mesh.type,mesh.ke[iel]) 

                for in=1:nen
                    djx[iel,ifac,in] =  dAi * N[iel,ifac,in] * VxDir[nodei]
                    djy[iel,ifac,in] =  dAi * N[iel,ifac,in] * VyDir[nodei]
                    rjx[iel,ifac,in] =  dAi * N[iel,ifac,in]
                    rjy[iel,ifac,in] =  dAi * N[iel,ifac,in]
                    for jn=1:nen
                        me[iel,in,jn] += pe[iel,ifac,jn] * taui * dAi * N[iel,ifac,in]
                    end
                end
            end
        end

        # Inverse of me matrix
        cf11 =  1.0; cf12 =-1.0; cf13 =  1.0;
        cf21 = -1.0; cf22 = 1.0; cf23 = -1.0;
        cf31 =  1.0; cf32 =-1.0; cf33 =  1.0;
        @tturbo for iel=1:mesh.nel 
            for ifac=1:mesh.nf_el
                Mm11 = me[iel,2,2] * me[iel,3,3] - me[iel,2,3] * me[iel,3,2]
                Mm12 = me[iel,2,1] * me[iel,3,3] - me[iel,2,3] * me[iel,3,1]
                Mm13 = me[iel,2,1] * me[iel,3,2] - me[iel,2,2] * me[iel,3,1]
                Mm21 = me[iel,1,2] * me[iel,3,3] - me[iel,1,3] * me[iel,3,2]
                Mm22 = me[iel,1,1] * me[iel,3,3] - me[iel,1,3] * me[iel,3,1]
                Mm23 = me[iel,1,1] * me[iel,3,2] - me[iel,1,2] * me[iel,3,1]
                Mm31 = me[iel,1,2] * me[iel,2,3] - me[iel,1,3] * me[iel,2,2]
                Mm32 = me[iel,1,1] * me[iel,2,3] - me[iel,1,3] * me[iel,2,1]
                Mm33 = me[iel,1,1] * me[iel,2,2] - me[iel,1,2] * me[iel,2,1]
                Ad11 = cf11*Mm11; Ad12 = cf21*Mm21; Ad13 = cf31*Mm31;
                Ad21 = cf12*Mm12; Ad22 = cf22*Mm22; Ad23 = cf32*Mm32;
                Ad31 = cf13*Mm13; Ad32 = cf23*Mm23; Ad33 = cf33*Mm33;
                DetM = me[iel,1,1] * Mm11 - me[iel,1,2]*Mm12 + me[iel,1,3]*Mm13
                mei[iel,1,1] = 1.0/DetM*Ad11; mei[iel,1,2] = 1.0/DetM*Ad12; mei[iel,1,3] = 1.0/DetM*Ad13;
                mei[iel,2,1] = 1.0/DetM*Ad21; mei[iel,2,2] = 1.0/DetM*Ad22; mei[iel,2,3] = 1.0/DetM*Ad23;
                mei[iel,3,1] = 1.0/DetM*Ad31; mei[iel,3,2] = 1.0/DetM*Ad32; mei[iel,3,3] = 1.0/DetM*Ad33;
            end
        end

        # Assembly of b vector
        @tturbo for iel=1:mesh.nel

            be_o2[iel,1,1] = mesh.vole[iel]*sex[iel]
            be_o2[iel,2,1] = mesh.vole[iel]*sey[iel]

            for ifac=1:mesh.nf_el
                    
                nodei = mesh.e2f[iel,ifac]
                bc    = mesh.bc[nodei]
                dAi   = mesh.dA[iel,ifac]
                taui  = StabParam(tau,dAi,mesh.vole[iel],mesh.type,mesh.ke[iel])                              # Stabilisation parameter for the face
                
                for in=1:nen
                    be_o2[iel,1,in] += (bc==1) * taui * djx[iel,ifac,in]
                    be_o2[iel,2,in] += (bc==1) * taui * djy[iel,ifac,in]
                end
                
            end
        end
    end
    return ae, be, be_o2, ze, pe, mei, pe, rjx, rjy

end

#--------------------------------------------------------------------#

function ComputeElementValues_o2(mesh, Vxh, Vyh, Pe, ae, be, be_o2, ze, rjx, rjy, mei, VxDir, VyDir, tau, o2)
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

    @tturbo for iel=1:mesh.nel
    
        Vxe[iel]  =  be[iel,1]/ae[iel]
        Vye[iel]  =  be[iel,2]/ae[iel]
        Txxe[iel] =  mesh.ke[iel]/mesh.vole[iel]*ze[iel,1,1]
        Tyye[iel] =  mesh.ke[iel]/mesh.vole[iel]*ze[iel,2,2] 
        Txye[iel] =  mesh.ke[iel]/mesh.vole[iel]*0.5*(ze[iel,1,2]+ze[iel,2,1])
        
        for ifac=1:mesh.nf_el
            
            # Face
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel],mesh.type,mesh.ke[iel])      # Stabilisation parameter for the face

            # First order
            O1 = (bc!=1) & (o2==0)
            Vxe[iel]  += (bc!=1) *  dAi*taui*Vxh[nodei]/ae[iel]
            Vye[iel]  += (bc!=1) *  dAi*taui*Vyh[nodei]/ae[iel]

            # Second order
            O2       = (bc!=1) & (o2==1)
            for in=1:nen
                Vxe1[iel,in] += O2 * taui*rjx[iel,ifac,in]*Vxh[nodei]
                Vye1[iel,in] += O2 * taui*rjy[iel,ifac,in]*Vyh[nodei]
            end

            # Assemble
            Txxe[iel] += (bc!=1) *  mesh.ke[iel]/mesh.vole[iel]*dAi*ni_x*Vxh[nodei]
            Tyye[iel] += (bc!=1) *  mesh.ke[iel]/mesh.vole[iel]*dAi*ni_y*Vyh[nodei]
            Txye[iel] += (bc!=1) *  mesh.ke[iel]*0.5*( 1.0/mesh.vole[iel]*dAi*( ni_x*Vyh[nodei] + ni_y*Vxh[nodei] ) )
        end
        Txxe[iel] *= 2.0
        Tyye[iel] *= 2.0
        Txye[iel] *= 2.0
        # inv(me) * ue
        Vxe2[iel,1] = (o2==1) * (mei[iel,1,1] * Vxe1[iel,1] + mei[iel,1,2] * Vxe1[iel,2] + mei[iel,1,3] * Vxe1[iel,3])
        Vxe2[iel,2] = (o2==1) * (mei[iel,2,1] * Vxe1[iel,1] + mei[iel,2,2] * Vxe1[iel,2] + mei[iel,2,3] * Vxe1[iel,3])
        Vxe2[iel,3] = (o2==1) * (mei[iel,3,1] * Vxe1[iel,1] + mei[iel,3,2] * Vxe1[iel,2] + mei[iel,3,3] * Vxe1[iel,3])
        # inv(me) * ue
        Vye2[iel,1] = (o2==1) * (mei[iel,1,1] * Vye1[iel,1] + mei[iel,1,2] * Vye1[iel,2] + mei[iel,1,3] * Vye1[iel,3])
        Vye2[iel,2] = (o2==1) * (mei[iel,2,1] * Vye1[iel,1] + mei[iel,2,2] * Vye1[iel,2] + mei[iel,2,3] * Vye1[iel,3])
        Vye2[iel,3] = (o2==1) * (mei[iel,3,1] * Vye1[iel,1] + mei[iel,3,2] * Vye1[iel,2] + mei[iel,3,3] * Vye1[iel,3])
    end

    if o2==1 Vxe .= Vxe2[:,1] end # extract first values of the ue solution vector (linear function evaluated at cell centroid)
    if o2==1 Vye .= Vye2[:,1] end # extract first values of the ue solution vector (linear function evaluated at cell centroid)

    return Vxe, Vye, Txxe, Tyye, Txye
end

#--------------------------------------------------------------------#

function ElementAssemblyLoop_o2(mesh, ae, be, be_o2, ze, mei, pe, rjx, rjy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, tau, o2)
    # Assemble element matrices and rhs
    f   = zeros(mesh.nf)
    Kuu = zeros(mesh.nel, 2*mesh.nf_el, 2*mesh.nf_el)
    fu  = zeros(mesh.nel, 2*mesh.nf_el)
    Kup = zeros(mesh.nel, 2*mesh.nf_el);
    fp  = zeros(mesh.nel)

    @tturbo for iel=1:mesh.nel 

        for ifac=1:mesh.nf_el 

            nodei = mesh.e2f[iel,ifac]
            bci   = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel], mesh.type,mesh.ke[iel])  
                
            for jfac=1:mesh.nf_el

                nodej = mesh.e2f[iel,jfac]
                bcj   = mesh.bc[nodej]   
                dAj   = mesh.dA[iel,jfac]
                nj_x  = mesh.n_x[iel,jfac]
                nj_y  = mesh.n_y[iel,jfac]
                tauj  = StabParam(tau,dAj,mesh.vole[iel], mesh.type,mesh.ke[iel])  
                        
                # Delta
                del = 0.0 + (ifac==jfac)*1.0
                        
                # Element matrix
                nitnj   = ni_x*nj_x + ni_y*nj_y;

                res     = 1.0/ae[iel] * dAj 
                # inv(me) * be_x
                meidx1  = mei[iel,1,1] * rjx[iel,jfac,1] + mei[iel,1,2] * rjx[iel,jfac,2] + mei[iel,1,3] * rjx[iel,jfac,3]
                meidx2  = mei[iel,2,1] * rjx[iel,jfac,1] + mei[iel,2,2] * rjx[iel,jfac,2] + mei[iel,2,3] * rjx[iel,jfac,3]
                meidx3  = mei[iel,3,1] * rjx[iel,jfac,1] + mei[iel,3,2] * rjx[iel,jfac,2] + mei[iel,3,3] * rjx[iel,jfac,3]
                rx      = (pe[iel,ifac,1] * meidx1 + pe[iel,ifac,2] * meidx2 + pe[iel,ifac,3] * meidx3)
                resx    = (o2==0) * res + (o2==1) * rx
                # inv(me) * be_y
                meidy1  = mei[iel,1,1] * rjy[iel,jfac,1] + mei[iel,1,2] * rjy[iel,jfac,2] + mei[iel,1,3] * rjy[iel,jfac,3]
                meidy2  = mei[iel,2,1] * rjy[iel,jfac,1] + mei[iel,2,2] * rjy[iel,jfac,2] + mei[iel,2,3] * rjy[iel,jfac,3]
                meidy3  = mei[iel,3,1] * rjy[iel,jfac,1] + mei[iel,3,2] * rjy[iel,jfac,2] + mei[iel,3,3] * rjy[iel,jfac,3]
                ry      = (pe[iel,ifac,1] * meidy1 + pe[iel,ifac,2] * meidy2 + pe[iel,ifac,3] * meidy3)
                resy    = (o2==0) * res + (o2==1) * ry

                Kexij =-dAi * (resx * taui*tauj - mesh.ke[iel]/mesh.vole[iel]*dAj*nitnj - taui*del)
                Keyij =-dAi * (resy * taui*tauj - mesh.ke[iel]/mesh.vole[iel]*dAj*nitnj - taui*del)
                yes   = (bci!=1) & (bcj!=1)
                Kuu[iel,ifac,           jfac           ] = yes * Kexij
                Kuu[iel,ifac+mesh.nf_el,jfac+mesh.nf_el] = yes * Keyij
                # Kuu[iel,ifac,           jfac           ] = (bci!=1) * (bcj!=1) * Ke_ij
                # Kuu[iel,ifac+mesh.nf_el,jfac+mesh.nf_el] = (bci!=1) * (bcj!=1) * Ke_ij
            end
            # RHS
            Xi      = 0.0 + (bci==2)*1.0
            Ji      = 0.0 + (bci==3)*1.0
            tix     = ni_x*SxxNeu[nodei] + ni_y*SxyNeu[nodei]
            tiy     = ni_x*SyxNeu[nodei] + ni_y*SyyNeu[nodei]
            nitze_x = ni_x*ze[iel,1,1] + ni_y*ze[iel,2,1]
            nitze_y = ni_x*ze[iel,1,2] + ni_y*ze[iel,2,2]
            #-----------------------------
            ax1    = 1.0/ae[iel]*be[iel,1] 
            # inv(me) * be_x
            meidx1  = mei[iel,1,1] * be_o2[iel,1,1] + mei[iel,1,2] * be_o2[iel,1,2] + mei[iel,1,3] * be_o2[iel,1,3]
            meidx2  = mei[iel,2,1] * be_o2[iel,1,1] + mei[iel,2,2] * be_o2[iel,1,2] + mei[iel,2,3] * be_o2[iel,1,3]
            meidx3  = mei[iel,3,1] * be_o2[iel,1,1] + mei[iel,3,2] * be_o2[iel,1,2] + mei[iel,3,3] * be_o2[iel,1,3]
            ax2    = (pe[iel,ifac,1] * meidx1 + pe[iel,ifac,2] * meidx2 + pe[iel,ifac,3] * meidx3)
            ax     = (o2==0) * ax1 + (o2==1) * ax2
            #-----------------------------
            ay1    = 1.0/ae[iel]*be[iel,2] 
            # inv(me) * be_y
            meidy1  = mei[iel,1,1] * be_o2[iel,2,1] + mei[iel,1,2] * be_o2[iel,2,2] + mei[iel,1,3] * be_o2[iel,2,3]
            meidy2  = mei[iel,2,1] * be_o2[iel,2,1] + mei[iel,2,2] * be_o2[iel,2,2] + mei[iel,2,3] * be_o2[iel,2,3]
            meidy3  = mei[iel,3,1] * be_o2[iel,2,1] + mei[iel,3,2] * be_o2[iel,2,2] + mei[iel,3,3] * be_o2[iel,2,3]
            ay2    = (pe[iel,ifac,1] * meidy1 + pe[iel,ifac,2] * meidy2 + pe[iel,ifac,3] * meidy3)
            ay     = (o2==0) * ay1 + (o2==1) * ay2
            #-----------------------------
            feix    = (bci!=1) * -dAi * (mesh.ke[iel]/mesh.vole[iel]*nitze_x - tix*Xi - gbar[iel,ifac,1]*Ji - ax*taui)
            feiy    = (bci!=1) * -dAi * (mesh.ke[iel]/mesh.vole[iel]*nitze_y - tiy*Xi - gbar[iel,ifac,2]*Ji - ay*taui)
            # up block
            Kup[iel,ifac]                            -= (bci!=1) * dAi*ni_x;
            Kup[iel,ifac+mesh.nf_el]                 -= (bci!=1) * dAi*ni_y;
            # Dirichlet nodes - uu block
            Kuu[iel,ifac,ifac]                       += (bci==1) * 1e6
            Kuu[iel,ifac+mesh.nf_el,ifac+mesh.nf_el] += (bci==1) * 1e6
            fu[iel,ifac]                             += (bci!=1) * feix + (bci==1) * VxDir[nodei] * 1e6
            fu[iel,ifac+mesh.nf_el]                  += (bci!=1) * feiy + (bci==1) * VyDir[nodei] * 1e6
            # Dirichlet nodes - pressure RHS
            fp[iel]                                  += (bci==1) * -dAi*(VxDir[nodei]*ni_x + VyDir[nodei]*ni_y)
        end
    end
    return Kuu, fu, Kup, fp
end

#--------------------------------------------------------------------#

function CreateTripletsSparse(mesh, Kuu_v, fu_v, Kup_v)
    # Create triplets and assemble sparse matrix fo Kuu
    e2fu = mesh.e2f
    e2fv = mesh.e2f .+ mesh.nf 
    e2f  = hcat(e2fu, e2fv)
    idof = 1:mesh.nf_el*2  
    ii   = repeat(idof, 1, length(idof))'
    ij   = repeat(idof, 1, length(idof))
    Ki   = e2f[:,ii]
    Kif  = e2f[:,ii[1,:]]
    Kj   = e2f[:,ij]
    Kuu  = sparse(Ki[:], Kj[:], Kuu_v[:], mesh.nf*2, mesh.nf*2)
    # file = matopen(string(@__DIR__,"/results/matrix_uu.mat"), "w" )
    # write(file, "Ki",       Ki[:] )
    # write(file, "Kj",    Kj[:] )
    # write(file, "Kuu",  Kuu_v[:] )
    # write(file, "nrow",  mesh.nf*2 )
    # write(file, "ncol",  mesh.nf*2 )
    # close(file)
    fu   = sparse(Kif[:], ones(size(Kif[:])), fu_v[:], mesh.nf*2, 1)
    fu   = Array(fu)
    droptol!(Kuu, 1e-6)
    # Create triplets and assemble sparse matrix fo Kup
    idof = 1:mesh.nf_el*2  
    ii   = repeat(idof, 1, mesh.nel)'
    ij   = repeat(1:mesh.nel, 1, length(idof))
    Ki   = e2f
    Kj   = ij
    Kup  = sparse(Ki[:], Kj[:], Kup_v[:], mesh.nf*2, mesh.nel  )
    # file = matopen(string(@__DIR__,"/results/matrix_up.mat"), "w" )
    # write(file, "Ki",       Ki[:] )
    # write(file, "Kj",    Kj[:] )
    # write(file, "Kup",  Kup_v[:] )
    # write(file, "nrow",  mesh.nf*2 )
    # write(file, "ncol",  mesh.nel )
    # close(file)
    return Kuu, fu, Kup
end

#--------------------------------------------------------------------#