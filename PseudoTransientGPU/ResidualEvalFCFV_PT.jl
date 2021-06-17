function ResidualOnFaces(mesh, Th, Te, qx, qy, tau) 
    F = zeros(mesh.nf)
    @avx for idof=1:mesh.nf 

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

#--------------------------------------------------------------------#
# bc
# yes1
# iel1
# yes2
# iel2
# function precomp(bc, yes1, yes2, iel1, iel2, mesh) 
#     # Threads.@threads for idof=1:mesh.nf 
#     for idof=1:mesh.nf 
#         # Identify BC faces
#         bc = mesh.bc[idof]
#         # Element 1
#         yes1  = mesh.f2e[idof,1]>0
#         iel1  = (yes1==1) * mesh.f2e[idof,1]  + (yes1==0) * 1 # if the neigbouring element is out then set index to 1
#         # Element 2
#         yes2  = mesh.f2e[idof,2]>0
#         iel2  = (yes2==1) * mesh.f2e[idof,2]   + (yes2==0) * 1 # if the neigbouring element is out then set index to 1
#     end
#     return
# end

function ResidualOnFaces_v2!(F, mesh, Th, Te, qx, qy, ae, be, ze, tau, Tneu) 
    # F  = zeros(mesh.nf)
    Threads.@threads for idof=1:mesh.nf 
    # for idof=1:mesh.nf 
        # Identify BC faces
        bc   = mesh.bc[idof]
        Xi   = 0.0 + (bc==2)*1.0;  # Neumann
        ti   = Tneu[idof]          # Neumann

        # Element 1
        yes1  = mesh.f2e[idof,1]>0
        iel1  = (yes1==1) * mesh.f2e[idof,1]  + (yes1==0) * 1 # if the neigbouring element is out then set index to 1
        dAi1  = (yes1==1) * mesh.dA_f[idof,1]  
        ni_x1 = (yes1==1) * mesh.n_x_f[idof,1] 
        ni_y1 = (yes1==1) * mesh.n_y_f[idof,1] 
        taui1 = (yes1==1) * StabParam(tau,dAi1,mesh.vole_f[idof,1],mesh.type)
        # >>>>>>>> Here, I assume Te, qx and qy are precomputed
        # Tel1  = (yes1==1) * Te[iel1] 
        # qxel1 = (yes1==1) * qx[iel1]
        # qyel1 = (yes1==1) * qy[iel1] 
        # >>>>>>>> Here they are computed on the fly
        Tel1   = (yes1==1) * be[iel1]/ae[iel1]
        qxel1  = (yes1==1) * -ze[iel1,1]/mesh.vole[iel1]
        qyel1  = (yes1==1) * -ze[iel1,2]/mesh.vole[iel1]
        for ifac=1:mesh.nf_el
            # Face
            nodei = mesh.e2f[iel1,ifac]
            bcf   = mesh.bc[nodei]
            dAi   = mesh.dA[iel1,ifac]
            ni_x  = mesh.n_x[iel1,ifac]
            ni_y  = mesh.n_y[iel1,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel1],mesh.type)      # Stabilisation parameter for the face
            # Assemble
            Tel1  += (yes1==1) * (bcf!=1) *  dAi*taui*Th[mesh.e2f[iel1, ifac]]/ae[iel1]
            qxel1 -= (yes1==1) * (bcf!=1) *  dAi*ni_x*Th[mesh.e2f[iel1, ifac]]/mesh.vole[iel1]
            qyel1 -= (yes1==1) * (bcf!=1) *  dAi*ni_y*Th[mesh.e2f[iel1, ifac]]/mesh.vole[iel1]
        end

        F1 = (bc!=1) * (yes1==1) * (dAi1*ni_x1*qxel1 + dAi1*ni_y1*qyel1 + dAi1*taui1*Tel1 - dAi1*taui1*Th[idof] + dAi1*Xi*ti)
        #######################################################

        # Element 2
        yes2  = mesh.f2e[idof,2]>0
        iel2  = (yes2==1) * mesh.f2e[idof,2]   + (yes2==0) * 1 # if the neigbouring element is out then set index to 1
        dAi2  = (yes2==1) * mesh.dA_f[idof,2]  
        ni_x2 = (yes2==1) * mesh.n_x_f[idof,2] 
        ni_y2 = (yes2==1) * mesh.n_y_f[idof,2]
        taui2 = (yes2==1) * StabParam(tau,dAi2,mesh.vole_f[idof,2],mesh.type) 
        # >>>>>>>> Here, I assume Te, qx and qy are precomputed
        # Tel2  = (yes2==1) * Te[iel2]
        # qxel2 = (yes2==1) * qx[iel2]
        # qyel2 = (yes2==1) * qy[iel2]
        # >>>>>>>> Here they are computed on the fly
        Tel2   = (yes2==1) * be[iel2]/ae[iel2]
        qxel2  = (yes2==1) * -ze[iel2,1]/mesh.vole[iel2]
        qyel2  = (yes2==1) * -ze[iel2,2]/mesh.vole[iel2]
        for ifac=1:mesh.nf_el
            # Face
            nodei = mesh.e2f[iel2,ifac]
            bcf    = mesh.bc[nodei]
            dAi   = mesh.dA[iel2,ifac]
            ni_x  = mesh.n_x[iel2,ifac]
            ni_y  = mesh.n_y[iel2,ifac]
            taui  = StabParam(tau,dAi,mesh.vole[iel2],mesh.type)      # Stabilisation parameter for the face
            # Assemble
            Tel2  += (yes2==1) * (bcf!=1) *  dAi*taui*Th[mesh.e2f[iel2, ifac]]/ae[iel2]
            qxel2 -= (yes2==1) * (bcf!=1) *  dAi*ni_x*Th[mesh.e2f[iel2, ifac]]/mesh.vole[iel2]
            qyel2 -= (yes2==1) * (bcf!=1) *  dAi*ni_y*Th[mesh.e2f[iel2, ifac]]/mesh.vole[iel2]
        end

        # nitze  = ni_x*ze[iel,1] + ni_y*ze[iel,2]
        # fe_i   = (bci!=1) * -dAi * (mesh.ke[iel]/mesh.vole[iel]*nitze - ti*Xi - 1.0/ae[iel]*be[iel]*taui)

        F2 = (bc!=1) * (yes2==1) * (dAi2*ni_x2*qxel2 + dAi2*ni_y2*qyel2 + dAi2*taui2*Tel2 - dAi2*taui2*Th[idof] + dAi2*Xi*ti)

        #######################################################
        
        # Contributions from the 2 elements
        F[idof] = F1 + F2
    end
    return F
end

#--------------------------------------------------------------------#

function ComputeElementValuesFaces(mesh, uh, ae, be, ze, Tdir, tau)
    ue          = zeros(mesh.nel);
    qx          = zeros(mesh.nel);
    qy          = zeros(mesh.nel);

    for idof=1:mesh.nf #

        # Identify BC faces
        bc = mesh.bc[idof]

        # Element 1
        yes1  = mesh.f2e[idof,1]>0
        iel1  = (yes1==1) * mesh.f2e[idof,1]  + (yes1==0) * 1 # if the neigbouring element is out then set index to 1
        dAi1  = (yes1==1) * mesh.dA_f[idof,1]  
        ni_x1 = (yes1==1) * mesh.n_x_f[idof,1] 
        ni_y1 = (yes1==1) * mesh.n_y_f[idof,1] 
        taui1 = (yes1==1) * StabParam(tau,dAi1,mesh.vole_f[idof,1],mesh.type) 

        ue[iel1] += (yes1==1) * 1.0/mesh.nf_el*be[iel1]/ae[iel1]
        ue[iel1] += (bc!=1) * (yes1==1) * dAi1*taui1*uh[idof]/ae[iel1]

        qx[iel1] += (yes1==1) * 1.0/mesh.nf_el* -1.0/mesh.vole[iel1]*ze[iel1,1]
        qx[iel1] -= (bc!=1) * (yes1==1) * 1.0/mesh.vole[iel1]*dAi1*ni_x1*uh[idof]

        qy[iel1] += (yes1==1) * 1.0/mesh.nf_el* -1.0/mesh.vole[iel1]*ze[iel1,2]
        qy[iel1] -= (bc!=1) * (yes1==1) * 1.0/mesh.vole[iel1]*dAi1*ni_y1*uh[idof]

        # Element 2
        yes2  = mesh.f2e[idof,2]>0
        iel2  = (yes2==1) * mesh.f2e[idof,2]   + (yes2==0) * 1 # if the neigbouring element is out then set index to 1
        dAi2  = (yes2==1) * mesh.dA_f[idof,2]  
        ni_x2 = (yes2==1) * mesh.n_x_f[idof,2] 
        ni_y2 = (yes2==1) * mesh.n_y_f[idof,2]
        taui2 = (yes2==1) * StabParam(tau,dAi2,mesh.vole_f[idof,2],mesh.type) 

        ue[iel2] += (yes2==1) * 1.0/mesh.nf_el*be[iel2]/ae[iel2]
        ue[iel2] += (bc!=1) * (yes2==1) * dAi2*taui2*uh[idof]/ae[iel2]
        
        qx[iel2] -= (yes2==1) * 1.0/mesh.nf_el* 1.0/mesh.vole[iel2]*ze[iel2,1]
        qx[iel2] -= (bc!=1) * (yes2==1) * 1.0/mesh.vole[iel2]*dAi2*ni_x2*uh[idof]

        qy[iel2] -= (yes2==1) * 1.0/mesh.nf_el* 1.0/mesh.vole[iel2]*ze[iel2,2]
        qy[iel2] -= (bc!=1) * (yes2==1) * 1.0/mesh.vole[iel2]*dAi2*ni_y2*uh[idof]

    end
    return ue, qx, qy
end

#--------------------------------------------------------------------#
