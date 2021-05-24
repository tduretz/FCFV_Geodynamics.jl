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
            taui  = StabParam(tau,dAi,mesh.vole[iel],mesh.type)                              # Stabilisation parameter for the face

            # Assemble
            ze[iel,1] += (bc==1) * dAi*ni_x*Tdir[nodei]  # Dirichlet
            ze[iel,2] += (bc==1) * dAi*ni_y*Tdir[nodei]  # Dirichlet
            be[iel]   += (bc==1) * dAi*taui*Tdir[nodei]  # Dirichlet
            ae[iel]   +=           dAi*taui
            
        end
    end
    return ae, be, ze
end

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

function ElementAssemblyLoop(mesh, ae, be, ze, Tdir, Tneu, tau)

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
                Ke_ij =-dAi * (1.0/ae[iel] * dAj * taui*tauj - 1.0/mesh.vole[iel]*dAj*nitnj - taui*del);
                Kv[iel,ifac,jfac] = (bci!=1) * (bcj!=1) * Ke_ij
            end
            # RHS
            Xi     = 0.0 + (bci==2)*1.0;
            ti     = Tneu[nodei]
            nitze  = ni_x*ze[iel,1] + ni_y*ze[iel,2]
            fe_i   = (bci!=1) * -dAi * (1.0/mesh.vole[iel]*nitze - ti*Xi - 1.0/ae[iel]*be[iel]*taui)
            fv[iel,ifac] += (bci!=1) * fe_i
            # Dirichlet nodes
            Kv[iel,ifac,ifac] += (bci==1) * 1.0
            fv[iel,ifac]      += (bci==1) * Tdir[nodei]
        end
    end
    return Kv, fv
end

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

function ResidualOnFaces(mesh, Th, Te, qx, qy, tau) 
    F = zeros(mesh.nf)
    @avx for idof=1:mesh.nf  # ne marche pas avec avx

        # Identify BC faces
        bc = mesh.bc[idof]

        # Element 1
        yes1  = mesh.f2e[idof,1]>0
        iel1  = (yes1==1) * mesh.f2e[idof,1]  + (yes1==0) * 1 # if the neigbouring element is out then set index to 1
        dAi1  = (yes1==1) * mesh.dA_f[idof,1]  
        ni_x1 = (yes1==1) * mesh.n_x_f[idof,1] 
        ni_y1 = (yes1==1) * mesh.n_y_f[idof,1] 
        taui1 = (yes1==1) * StabParam(tau,dAi1,mesh.vole_f[idof,1],mesh.type)  
        Tel1  = (yes1==1) * Te[iel1] 
        qxel1 = (yes1==1) * qx[iel1]
        qyel1 = (yes1==1) * qy[iel1] 
        F1    = (bc!=1) * (yes1==1) * (dAi1*ni_x1*qxel1 + dAi1*ni_y1*qyel1 + dAi1*taui1*Tel1 - dAi1*taui1*Th[idof])

        # Element 2
        yes2  = mesh.f2e[idof,2]>0
        iel2  = (yes2==1) * mesh.f2e[idof,2]   + (yes2==0) * 1 # if the neigbouring element is out then set index to 1
        dAi2  = (yes2==1) * mesh.dA_f[idof,2]  
        ni_x2 = (yes2==1) * mesh.n_x_f[idof,2] 
        ni_y2 = (yes2==1) * mesh.n_y_f[idof,2]
        taui2 = (yes2==1) * StabParam(tau,dAi2,mesh.vole_f[idof,2],mesh.type) 
        Tel2  = (yes2==1) * Te[iel2]
        qxel2 = (yes2==1) * qx[iel2]
        qyel2 = (yes2==1) * qy[iel2]
        F2    = (bc!=1) * (yes2==1) * (dAi2*ni_x2*qxel2 + dAi2*ni_y2*qyel2 + dAi2*taui2*Tel2 - dAi2*taui2*Th[idof])
        
        # Contributions from the 2 elements
        F[idof] = F1 + F2
    end
    return F 
end

# function ResidualOnFaces(mesh, Th, Te, qx, qy, tau) 
#     F = zeros(mesh.nf)
#     for idof=1:mesh.nf  # ne marche pas avec avx

#         bc = mesh.bc[idof]

#         # element 1
#         iel   = mesh.f2e[idof,1]
#         yes   = iel>0
#         dAi   = (yes==1) * mesh.dA_f[idof,1]  + (yes==0) * 0.0
#         ni_x  = (yes==1) * mesh.n_x_f[idof,1] + (yes==0) * 0.0
#         ni_y  = (yes==1) * mesh.n_y_f[idof,1] + (yes==0) * 0.0
#         tau0   = StabParam(tau,dAi,mesh.vole_f[idof,1],mesh.type) 
#         taui  = (yes==1) * tau0 + (yes==0) * 0.0
#         Tel   = (yes==1) * Te[iel] + (yes==0) * 0.0
#         qxel  = (yes==1) * qx[iel] + (yes==0) * 0.0
#         qyel  = (yes==1) * qy[iel] + (yes==0) * 0.0
        
#         F1 = (bc!=1) * (yes==1) * (dAi*ni_x*qxel + dAi*ni_y*qyel + dAi*taui*Tel - dAi*taui*Th[idof])

#         # element 2
#         iel   = mesh.f2e[idof,2]
#         yes   = iel>0
#         # if iel>0
#         #     dAi   = (yes==1) * mesh.dA_f[idof,2]  + (yes==0) * 0.0
#         #     ni_x  = (yes==1) * mesh.n_x_f[idof,2] + (yes==0) * 0.0
#         #     ni_y  = (yes==1) * mesh.n_y_f[idof,2] + (yes==0) * 0.0
#         #     tau0   = StabParam(tau,dAi,mesh.vole_f[idof,2],mesh.type) 
#         #     taui  = (yes==1) * tau0 + (yes==0) * 0.0
#         #     Tel   = (yes==1) * Te[iel] + (yes==0) * 0.0
#         #     qxel  = (yes==1) * qx[iel] + (yes==0) * 0.0
#         #     qyel  = (yes==1) * qy[iel] + (yes==0) * 0.0
#         #     F[idof] += (bc!=1) * (yes==1) * (dAi*ni_x*qxel + dAi*ni_y*qyel + dAi*taui*Tel - dAi*taui*Th[idof])
#         # end
#         iel   = (yes==1) * iel  + (yes==0) * 1
#         dAi   = (yes==1) * mesh.dA_f[idof,2]  + (yes==0) * 0.0
#         ni_x  = (yes==1) * mesh.n_x_f[idof,2] + (yes==0) * 0.0
#         ni_y  = (yes==1) * mesh.n_y_f[idof,2] + (yes==0) * 0.0
#         tau0  = StabParam(tau,dAi,mesh.vole_f[idof,2],mesh.type) 
#         taui  = (yes==1) * tau0 + (yes==0) * 0.0
#         Tel   = (yes==1) * Te[iel] + (yes==0) * 0.0
#         qxel  = (yes==1) * qx[iel] + (yes==0) * 0.0
#         qyel  = (yes==1) * qy[iel] + (yes==0) * 0.0
#         F2      = (bc!=1) * (yes==1) * (dAi*ni_x*qxel + dAi*ni_y*qyel + dAi*taui*Tel - dAi*taui*Th[idof])
#         F[idof] = F1 + F2
#     end
#     return F 
# end