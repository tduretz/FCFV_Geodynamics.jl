function ComputeFCFV(mesh, se, Tdir)
    ae = zeros(mesh.nel)
    be = zeros(mesh.nel)
    ze = zeros(mesh.nel,2)

    # Assemble FCFV elements
    @avx for iel=1:mesh.nel  

        be[iel] = be[iel]   + mesh.Ω[iel]*se[iel]
        
        for ifac=1:mesh.nf_el
            
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            Γi    = mesh.Γ[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            τi    = mesh.τ[nodei]      

            # Assemble
            ze[iel,1] += (bc==1) * Γi*ni_x*Tdir[nodei]  # Dirichlet
            ze[iel,2] += (bc==1) * Γi*ni_y*Tdir[nodei]  # Dirichlet
            be[iel]   += (bc==1) * Γi*τi*Tdir[nodei]    # Dirichlet
            ae[iel]   +=           Γi*τi
            
        end
    end
    return ae, be, ze
end

#--------------------------------------------------------------------#

function ComputeElementValues(mesh, uh, ae, be, ze, Tdir)
    ue          = zeros(mesh.nel);
    qx          = zeros(mesh.nel);
    qy          = zeros(mesh.nel);

    @avx for iel=1:mesh.nel
    
        ue[iel]  =  be[iel]/ae[iel]
        qx[iel]  = -mesh.ke[iel]/mesh.Ω[iel]*ze[iel,1]
        qy[iel]  = -mesh.ke[iel]/mesh.Ω[iel]*ze[iel,2]
        
        for ifac=1:mesh.nf_el
            
            # Face
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            Γi    = mesh.Γ[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            τi    = mesh.τ[nodei]

            # Assemble
            ue[iel] += (bc!=1) *  Γi*τi*uh[mesh.e2f[iel, ifac]]/ae[iel]
            qx[iel] -= (bc!=1) *  mesh.ke[iel]/mesh.Ω[iel]*Γi*ni_x*uh[mesh.e2f[iel, ifac]]
            qy[iel] -= (bc!=1) *  mesh.ke[iel]/mesh.Ω[iel]*Γi*ni_y*uh[mesh.e2f[iel, ifac]]
         end
    end
    return ue, qx, qy
end

#--------------------------------------------------------------------#

function ElementAssemblyLoop(mesh, ae, be, ze, Tdir, Tneu)

    # Assemble element matrices and rhs
    Kv   = zeros(mesh.nel, mesh.nf_el, mesh.nf_el)
    fv   = zeros(mesh.nel, mesh.nf_el)

    @avx for iel=1:mesh.nel 

        for ifac=1:mesh.nf_el 

            nodei = mesh.e2f[iel,ifac]
            bci   = mesh.bc[nodei]
            Γi    = mesh.Γ[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            τi    = mesh.τ[nodei] 
                
            for jfac=1:mesh.nf_el

                nodej = mesh.e2f[iel,jfac]
                bcj   = mesh.bc[nodej]   
                Γj    = mesh.Γ[iel,jfac]
                nj_x  = mesh.n_x[iel,jfac]
                nj_y  = mesh.n_y[iel,jfac]
                τj    = mesh.τ[nodej]
                        
                # Delta
                δ = 0.0 + (ifac==jfac)*1.0
                        
                # Element matrix
                nitnj = ni_x*nj_x + ni_y*nj_y;
                Ke_ij =-Γi * (1.0/ae[iel] * Γj * τi*τj - mesh.ke[iel]/mesh.Ω[iel]*Γj*nitnj - τi*δ);
                yes   = (bci!=1) & (bcj!=1)
                Kv[iel,ifac,jfac] = yes * Ke_ij
            end
            # RHS
            Xi     = 0.0 + (bci==2)*1.0;
            ti     = Tneu[nodei]
            nitze  = ni_x*ze[iel,1] + ni_y*ze[iel,2]
            fe_i   = (bci!=1) * -Γi * (mesh.ke[iel]/mesh.Ω[iel]*nitze - ti*Xi - 1.0/ae[iel]*be[iel]*τi)
            fv[iel,ifac] += (bci!=1) * fe_i
            # Dirichlet nodes
            Kv[iel,ifac,ifac] += (bci==1) * 1.0
            fv[iel,ifac]      += (bci==1) * Tdir[nodei]
        end
    end
    return Kv, fv
end

#--------------------------------------------------------------------#

function CreateTripletsSparse(mesh, Kv, fv)
    # Create triplets and assemble sparse matrix
    idof = 1:mesh.nf_el  
    ii   = repeat(idof, 1, length(idof))'
    ij   = repeat(idof, 1, length(idof))
    Ki   = mesh.e2f[:,ii]
    Kif  = mesh.e2f[:,ii[1,:]]
    Kj   = mesh.e2f[:,ij]
    K    = sparse(Ki[:], Kj[:], Kv[:], mesh.nf, mesh.nf)
    f    = sparse(Kif[:], ones(size(Kif[:])), fv[:], mesh.nf, 1)
    f    = Array(f)
    droptol!(K, 1e-6)
    return K, f
end

#--------------------------------------------------------------------#

function PoissonResidual(Rh, mesh, Th, Te, qx, qy)

    # Compute residual of global equation
    for iel=1:mesh.nel  
        
        for ifac=1:mesh.nf_el
            
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            Γi    = mesh.Γ[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            τi    = mesh.τ[nodei]      

            # Assemble
            Rh[nodei] += (bc==0) * Γi * ((ni_x*qx[iel] + ni_y*qy[iel]) + τi*(Te[iel] - Th[nodei]) ) 
            # Rh[nodei] += (bc==0) * Γi * ( τi*(Te[iel] - Th[nodei])) 


        end
    end
end

 #--------------------------------------------------------------------#