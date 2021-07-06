function ComputeFCFV_o2(mesh, se, Tdir, tau, o2)
    nen   = 3
    ae    = zeros(mesh.nel)
    be    = zeros(mesh.nel)     
    be_o2 = zeros(mesh.nel,nen) 
    ze    = zeros(mesh.nel,2)
    pe    = zeros(mesh.nel,mesh.nf_el,nen)
    me    = zeros(mesh.nel,nen,nen)
    mei   = zeros(mesh.nel,nen,nen)
    N     = zeros(mesh.nel,mesh.nf_el,nen)
    dj    = zeros(mesh.nel,mesh.nf_el,nen)
    rj    = zeros(mesh.nel,mesh.nf_el,nen)

    @tturbo for iel=1:mesh.nel  

        be[iel] = be[iel] + (o2==0)*mesh.vole[iel]*se[iel] 
        
        for ifac=1:mesh.nf_el
            
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau*mesh.ke[iel],dAi,mesh.vole[iel],mesh.type)                              # Stabilisation parameter for the face

            N[iel,ifac,1] = 1.0
            N[iel,ifac,2] = mesh.xf[nodei] - mesh.xc[iel]
            N[iel,ifac,3] = mesh.yf[nodei] - mesh.yc[iel]

            pe[iel,ifac,1] = 1.0
            pe[iel,ifac,2] = mesh.xf[nodei] - mesh.xc[iel]
            pe[iel,ifac,3] = mesh.yf[nodei] - mesh.yc[iel]
            
            # Assemble
            ze[iel,1] += (bc==1) * dAi*ni_x*Tdir[nodei]  # Dirichlet
            ze[iel,2] += (bc==1) * dAi*ni_y*Tdir[nodei]  # Dirichlet
            ae[iel]   +=           dAi*taui
            be[iel]   += (bc==1) *  (o2==0) * dAi*taui*Tdir[nodei]  # Dirichlet

        end
    end

    if o2==1 # The remaining of this function is only done if second order FCFV is used

        # me matrix
        @tturbo for iel=1:mesh.nel 
            for ifac=1:mesh.nf_el

                nodei = mesh.e2f[iel,ifac]
                bc    = mesh.bc[nodei]
                dAi   = mesh.dA[iel,ifac] 
                taui  = StabParam(tau*mesh.ke[iel],dAi,mesh.vole[iel],mesh.type)

                for in=1:nen
                    dj[iel,ifac,in] =  dAi * N[iel,ifac,in] * Tdir[nodei]
                    rj[iel,ifac,in] =  dAi * N[iel,ifac,in]
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

            be_o2[iel,1] = mesh.vole[iel]*se[iel]

            for ifac=1:mesh.nf_el
                    
                nodei = mesh.e2f[iel,ifac]
                bc    = mesh.bc[nodei]
                dAi   = mesh.dA[iel,ifac]
                taui  = StabParam(tau*mesh.ke[iel],dAi,mesh.vole[iel],mesh.type)                              # Stabilisation parameter for the face
                
                for in=1:nen
                    be_o2[iel,in] += (bc==1) * taui * dj[iel,ifac,in]
                end
                
            end
        end
    end
    return ae, be, be_o2, ze, pe, mei, pe, rj
end

#--------------------------------------------------------------------#

function ComputeElementValues_o2(mesh, uh, ae, be, be_o2, ze, rj, mei, Tdir, tau, o2)
    nen         = 3
    ue          = zeros(mesh.nel)
    ue1         = zeros(mesh.nel, nen);
    ue2         = zeros(mesh.nel, nen);
    qx          = zeros(mesh.nel);
    qy          = zeros(mesh.nel);

    @tturbo for iel=1:mesh.nel
    
        ue[iel]  =  (o2==0) * be[iel]/ae[iel]
        for in=1:nen
            ue1[iel,in]  =  (o2==1) * be_o2[iel,in]
        end
        
        qx[iel]  = -mesh.ke[iel]/mesh.vole[iel]*ze[iel,1]
        qy[iel]  = -mesh.ke[iel]/mesh.vole[iel]*ze[iel,2]
        
        for ifac=1:mesh.nf_el
            
            # Face
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau*mesh.ke[iel],dAi,mesh.vole[iel],mesh.type)      # Stabilisation parameter for the face

            # First order
            O1 = (bc!=1) & (o2==0)
            ue[iel] += O1 * dAi*taui*uh[mesh.e2f[iel, ifac]]/ae[iel]
            # Second order
            O2       = (bc!=1) & (o2==1)
            for in=1:nen
                ue1[iel,in] += O2 * taui*rj[iel,ifac,in]*uh[mesh.e2f[iel, ifac]]
            end
            qx[iel] -= (bc!=1) *  mesh.ke[iel]/mesh.vole[iel]*dAi*ni_x*uh[mesh.e2f[iel, ifac]]
            qy[iel] -= (bc!=1) *  mesh.ke[iel]/mesh.vole[iel]*dAi*ni_y*uh[mesh.e2f[iel, ifac]]
        end
        # inv(me) * ue
        ue2[iel,1] = (o2==1) * (mei[iel,1,1] * ue1[iel,1] + mei[iel,1,2] * ue1[iel,2] + mei[iel,1,3] * ue1[iel,3])
        ue2[iel,2] = (o2==1) * (mei[iel,2,1] * ue1[iel,1] + mei[iel,2,2] * ue1[iel,2] + mei[iel,2,3] * ue1[iel,3])
        ue2[iel,3] = (o2==1) * (mei[iel,3,1] * ue1[iel,1] + mei[iel,3,2] * ue1[iel,2] + mei[iel,3,3] * ue1[iel,3])
    end

    if o2==1 ue .= ue2[:,1] end # extract first values of the ue solution vector (linear function evaluated at cell centroid)
    
    return ue, qx, qy
end

#--------------------------------------------------------------------#

function ElementAssemblyLoop_o2(mesh, ae, be, be_o2, ze, mei, pe, rj, Tdir, Tneu, tau, o2)
    # Assemble element matrices and rhs
    f    = zeros(mesh.nf);
    Kv   = zeros(mesh.nel, mesh.nf_el, mesh.nf_el)
    fv   = zeros(mesh.nel, mesh.nf_el)

    @tturbo for iel=1:mesh.nel

        for ifac=1:mesh.nf_el 

            nodei = mesh.e2f[iel,ifac]
            bci   = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau*mesh.ke[iel],dAi,mesh.vole[iel], mesh.type)  
                
            for jfac=1:mesh.nf_el

                nodej = mesh.e2f[iel,jfac]
                bcj   = mesh.bc[nodej]   
                dAj   = mesh.dA[iel,jfac]
                nj_x  = mesh.n_x[iel,jfac]
                nj_y  = mesh.n_y[iel,jfac]
                tauj  = StabParam(tau*mesh.ke[iel],dAj,mesh.vole[iel], mesh.type)  
                        
                # Delta
                del = 0.0 + (ifac==jfac)*1.0
                        
                # Element matrix
                nitnj = ni_x*nj_x + ni_y*nj_y;
                res1  = 1.0/ae[iel] * dAj
                # inv(me) * dj
                meid1 = mei[iel,1,1] * rj[iel,jfac,1] + mei[iel,1,2] * rj[iel,jfac,2] + mei[iel,1,3] * rj[iel,jfac,3]
                meid2 = mei[iel,2,1] * rj[iel,jfac,1] + mei[iel,2,2] * rj[iel,jfac,2] + mei[iel,2,3] * rj[iel,jfac,3]
                meid3 = mei[iel,3,1] * rj[iel,jfac,1] + mei[iel,3,2] * rj[iel,jfac,2] + mei[iel,3,3] * rj[iel,jfac,3]
                # pe' * inv(me) * dj
                res2   =  (pe[iel,ifac,1] * meid1 + pe[iel,ifac,2] * meid2 + pe[iel,ifac,3] * meid3)
                res   = (o2==0) * res1 + (o2==1) * res2
                Ke_ij =-dAi * (res * taui*tauj - mesh.ke[iel]/mesh.vole[iel]*dAj*nitnj - taui*del);
                yes   = (bci!=1) & (bcj!=1)
                Kv[iel,ifac,jfac] = yes * Ke_ij
            end
            # RHS
            Xi     = 0.0 + (bci==2)*1.0;
            ti     = Tneu[nodei]
            nitze  = ni_x*ze[iel,1] + ni_y*ze[iel,2]
            res1    = 1.0/ae[iel]*be[iel] 
            # inv(me) * be
            meid1  = mei[iel,1,1] * be_o2[iel,1] + mei[iel,1,2] * be_o2[iel,2] + mei[iel,1,3] * be_o2[iel,3]
            meid2  = mei[iel,2,1] * be_o2[iel,1] + mei[iel,2,2] * be_o2[iel,2] + mei[iel,2,3] * be_o2[iel,3]
            meid3  = mei[iel,3,1] * be_o2[iel,1] + mei[iel,3,2] * be_o2[iel,2] + mei[iel,3,3] * be_o2[iel,3]
            res2   = (pe[iel,ifac,1] * meid1 + pe[iel,ifac,2] * meid2 + pe[iel,ifac,3] * meid3)
            res    = (o2==0) * res1 + (o2==1) * res2
            fe_i   = (bci!=1) * -dAi * (mesh.ke[iel]/mesh.vole[iel]*nitze - ti*Xi - res*taui)
            fv[iel,ifac] += (bci!=1) * fe_i
            # Dirichlet nodes
            Kv[iel,ifac,ifac] += (bci==1) * 1.0
            fv[iel,ifac]      += (bci==1) * Tdir[nodei]
        end
        # if iel==1
        #     println(Kv[iel,:,:])
        #     println(fv[iel,:])
        # end
    end
    return Kv, fv
end