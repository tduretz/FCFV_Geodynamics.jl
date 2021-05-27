function ComputeFCFV(mesh, sex, sey, VxDir, VxNeu, VyDir, VyNeu, tau)
    ae = zeros(mesh.nel)
    be = zeros(mesh.nel,2)
    ze = zeros(mesh.nel,2,2)

    # Assemble FCFV elements
    @avx for iel=1:mesh.nel  

        be[iel,1] += mesh.vole[iel]*sex[iel]
        be[iel,2] += mesh.vole[iel]*sey[iel]
        
        for ifac=1:mesh.nf_el
            
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel],mesh.type)                              # Stabilisation parameter for the face

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
    return ae, be, ze
end

#--------------------------------------------------------------------#

function ComputeElementValues(mesh, uh, ae, be, ze, Tdir, tau)
    ue          = zeros(mesh.nel);
    qx          = zeros(mesh.nel);
    qy          = zeros(mesh.nel);

    @avx for iel=1:mesh.nel
    
        ue[iel]  =  be[iel]/ae[iel]
        qx[iel]  = -1.0/mesh.vole[iel]*ze[iel,1]
        qy[iel]  = -1.0/mesh.vole[iel]*ze[iel,2]
        
        for ifac=1:mesh.nf_el
            
            # Face
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel],mesh.type)      # Stabilisation parameter for the face

            # Assemble
            ue[iel] += (bc!=1) *  dAi*taui*uh[mesh.e2f[iel, ifac]]/ae[iel]
            qx[iel] -= (bc!=1) *  1.0/mesh.vole[iel]*dAi*ni_x*uh[mesh.e2f[iel, ifac]]
            qy[iel] -= (bc!=1) *  1.0/mesh.vole[iel]*dAi*ni_y*uh[mesh.e2f[iel, ifac]]
         end
    end
    return ue, qx, qy
end

#--------------------------------------------------------------------#

function ElementAssemblyLoop(mesh, ae, be, ze, VxDir, VxNeu, VyDir, VyNeu, tau)
    # Assemble element matrices and rhs
    f   = zeros(mesh.nf);
    Kuu = zeros(mesh.nel, 2*mesh.nf_el, 2*mesh.nf_el)
    fu  = zeros(mesh.nel, 2*mesh.nf_el)
    Kup = zeros(mesh.nel, 2*mesh.nf_el);
    fp  = zeros(mesh.nel)
    I11 = 1.0
    I22 = 1.0

    @avx for iel=1:mesh.nel 

        for ifac=1:mesh.nf_el 

            nodei = mesh.e2f[iel,ifac]
            bci   = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel], mesh.type)  
                
            for jfac=1:mesh.nf_el

                nodej = mesh.e2f[iel,jfac]
                bcj   = mesh.bc[nodej]   
                dAj   = mesh.dA[iel,jfac]
                nj_x  = mesh.n_x[iel,jfac]
                nj_y  = mesh.n_y[iel,jfac]
                tauj  = StabParam(tau,dAj,mesh.vole[iel], mesh.type)  
                        
                # Delta
                del = 0.0 + (ifac==jfac)*1.0
                        
                # Element matrix
                nitnj = ni_x*nj_x + ni_y*nj_y;
                Ke_ij =-dAi * (1.0/ae[iel] * dAj * taui*tauj - 1.0/mesh.vole[iel]*dAj*nitnj - taui*del)
                Kuu[iel,ifac,           jfac           ] = (bci!=1) * (bcj!=1) * Ke_ij
                Kuu[iel,ifac+mesh.nf_el,jfac+mesh.nf_el] = (bci!=1) * (bcj!=1) * Ke_ij
            end
            # RHS
            Xi      = 0.0 + (bci==2)*1.0;
            tix     = VxNeu[nodei]
            tiy     = VyNeu[nodei]
            nitze_x = ni_x*ze[iel,1,1] + ni_y*ze[iel,2,1]
            nitze_y = ni_x*ze[iel,1,2] + ni_y*ze[iel,2,2]
            feix    = (bci!=1) * -dAi * (1.0/mesh.vole[iel]*nitze_x - tix*Xi - 1.0/ae[iel]*be[iel,1]*taui)
            feix    = (bci!=1) * -dAi * (1.0/mesh.vole[iel]*nitze_y - tiy*Xi - 1.0/ae[iel]*be[iel,2]*taui)
            # velocity RHS
            fu[iel,ifac           ]                  += (bci!=1) * feix
            fu[iel,ifac+mesh.nf_el]                  += (bci!=1) * feix
            # up block
            Kup[iel,ifac]                            -= (bci!=1) * dAi*ni_x;
            Kup[iel,ifac+mesh.nf_el]                 -= (bci!=1) * dAi*ni_y;
            # Dirichlet nodes - uu block
            Kuu[iel,ifac,ifac]                       += (bci==1) * 1.0
            Kuu[iel,ifac+mesh.nf_el,ifac+mesh.nf_el] += (bci==1) * 1.0
            fu[iel,ifac]                             += (bci==1) * VxDir[nodei]
            fu[iel,ifac+mesh.nf_el]                  += (bci==1) * VyDir[nodei]
            # Dirichlet nodes - pressure RHS
            fp[iel]                                  += (bci==1) * dAi*(VxDir[nodei]*ni_x + VyDir[nodei]*ni_y)
        end
    end
    return Kuu, fu, Kup, fp
end

#--------------------------------------------------------------------#

function CreateTripletsSparse(mesh, Kuu_v, fu_v, Kup_v)
    # Create triplets and assemble sparse matrix
    e2fu = mesh.e2f
    e2fv = mesh.e2f .+ mesh.nf 
    e2f  = hcat(e2fu, e2fv)
    println(size(e2f))
    idof = 1:mesh.nf_el*2  
    ii   = repeat(idof, 1, length(idof))'
    ij   = repeat(idof, 1, length(idof))
    Ki   = e2f[:,ii]
    Kif  = e2f[:,ii[1,:]]
    Kj   = e2f[:,ij]
    Kuu  = sparse(Ki[:], Kj[:], Kuu_v[:], mesh.nf*2, mesh.nf*2)
    fu   = sparse(Kif[:], ones(size(Kif[:])), fu_v[:], mesh.nf*2, 1)
    fu   = Array(fu)
    droptol!(Kuu, 1e-6)

    idof = 1:mesh.nf_el*2  
    ii   = repeat(idof, 1, mesh.nel)'
    ij   = repeat(1:mesh.nel, 1, length(idof))
    Ki   = e2f
    Kj   = ij
    Kup  = sparse(Ki[:], Kj[:], Kup_v[:], mesh.nf*2, mesh.nel  )
    return Kuu, fu, Kup
end

#--------------------------------------------------------------------#