function ComputeFCFV(mesh, se, Tdir, tau)
    ae = zeros(mesh.nel)
    be = zeros(mesh.nel)
    ze = zeros(mesh.nel,2)

    # Assemble FCFV elements
    @avx for iel=1:mesh.nel  

        be[iel] = be[iel]   + mesh.vole[iel]*se[iel]
        
        for ifac=1:mesh.nf_el
            
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau*mesh.ke[iel],dAi,mesh.vole[iel],mesh.type)                              # Stabilisation parameter for the face

            # Assemble
            ze[iel,1] += (bc==1) * dAi*ni_x*Tdir[nodei]  # Dirichlet
            ze[iel,2] += (bc==1) * dAi*ni_y*Tdir[nodei]  # Dirichlet
            be[iel]   += (bc==1) * dAi*taui*Tdir[nodei]  # Dirichlet
            ae[iel]   +=           dAi*taui
            
        end
    end
    return ae, be, ze
end

#--------------------------------------------------------------------#

function ComputeElementValues(mesh, uh, ae, be, ze, Tdir, tau)
    ue          = zeros(mesh.nel);
    qx          = zeros(mesh.nel);
    qy          = zeros(mesh.nel);

    @avx for iel=1:mesh.nel
    
        ue[iel]  =  be[iel]/ae[iel]
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

            # Assemble
            ue[iel] += (bc!=1) *  dAi*taui*uh[mesh.e2f[iel, ifac]]/ae[iel]
            qx[iel] -= (bc!=1) *  mesh.ke[iel]/mesh.vole[iel]*dAi*ni_x*uh[mesh.e2f[iel, ifac]]
            qy[iel] -= (bc!=1) *  mesh.ke[iel]/mesh.vole[iel]*dAi*ni_y*uh[mesh.e2f[iel, ifac]]
         end
    end
    return ue, qx, qy
end

#--------------------------------------------------------------------#

function ElementAssemblyLoop(mesh, ae, be, ze, Tdir, Tneu, tau)
    # Assemble element matrices and rhs
    f    = zeros(mesh.nf);
    Kv   = zeros(mesh.nel, mesh.nf_el, mesh.nf_el)
    fv   = zeros(mesh.nel, mesh.nf_el)

    @avx for iel=1:mesh.nel 

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
                Ke_ij =-dAi * (1.0/ae[iel] * dAj * taui*tauj - mesh.ke[iel]/mesh.vole[iel]*dAj*nitnj - taui*del);
                Kv[iel,ifac,jfac] = (bci!=1) * (bcj!=1) * Ke_ij
            end
            # RHS
            Xi     = 0.0 + (bci==2)*1.0;
            ti     = Tneu[nodei]
            nitze  = ni_x*ze[iel,1] + ni_y*ze[iel,2]
            fe_i   = (bci!=1) * -dAi * (mesh.ke[iel]/mesh.vole[iel]*nitze - ti*Xi - 1.0/ae[iel]*be[iel]*taui)
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