function ResidualOnFaces(mesh, Th, Te, qx, qy) 
    F = zeros(mesh.nf)
    @avx for idof=1:mesh.nf 

        # Identify BC faces
        bc = mesh.bc[idof]

        # Element 1
        yes1  = mesh.f2e[idof,1]>0
        iel1  = (yes1==1) * mesh.f2e[idof,1]  + (yes1==0) * 1 # if the neigbouring element is out then set index to 1
        Γ1    = (yes1==1) * mesh.Γ_f[idof,1]  
        ni_x1 = (yes1==1) * mesh.n_x_f[idof,1] 
        ni_y1 = (yes1==1) * mesh.n_y_f[idof,1] 
        τ1    = (yes1==1) * mesh.τ[idof]  
        Tel1  = (yes1==1) * Te[iel1] 
        qxel1 = (yes1==1) * qx[iel1]
        qyel1 = (yes1==1) * qy[iel1] 
        F1    = (bc!=1) * (yes1==1) * Γ1*(ni_x1*qxel1 + ni_y1*qyel1 + τ1*(Tel1 - Th[idof]))

        # Element 2
        yes2  = mesh.f2e[idof,2]>0
        iel2  = (yes2==1) * mesh.f2e[idof,2]   + (yes2==0) * 1 # if the neigbouring element is out then set index to 1
        Γ2    = (yes2==1) * mesh.Γ_f[idof,2]  
        ni_x2 = (yes2==1) * mesh.n_x_f[idof,2] 
        ni_y2 = (yes2==1) * mesh.n_y_f[idof,2]
        τ2    = (yes2==1) * mesh.τ[idof]
        Tel2  = (yes2==1) * Te[iel2]
        qxel2 = (yes2==1) * qx[iel2]
        qyel2 = (yes2==1) * qy[iel2]
        F2    = (bc!=1) * (yes2==1) * Γ2*(ni_x2*qxel2 + ni_y2*qyel2 + τ2*(Tel2 - Th[idof]))
        
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

function ResidualOnFaces_v2!(F, mesh, Th, ae, be, ze, Tneu) 
    # F  = zeros(mesh.nf)
    @inbounds Threads.@threads for idof=1:mesh.nf 
    # for idof=1:mesh.nf 
        # Identify BC faces
        bc   = mesh.bc[idof]
        Xi   = 0.0 + (bc==2)*1.0;  # Neumann
        ti   = Tneu[idof]          # Neumann

        # Element 1
        yes1  = mesh.f2e[idof,1]>0
        iel1  = (yes1==1) * mesh.f2e[idof,1]  + (yes1==0) * 1 # if the neigbouring element is out then set index to 1
        Γ1    = (yes1==1) * mesh.Γ_f[idof,1]  
        ni_x1 = (yes1==1) * mesh.n_x_f[idof,1] 
        ni_y1 = (yes1==1) * mesh.n_y_f[idof,1] 
        τ1    = (yes1==1) * mesh.τ[idof]
        # >>>>>>>> Here, I assume Te, qx and qy are precomputed
        # Tel1  = (yes1==1) * Te[iel1] 
        # qxel1 = (yes1==1) * qx[iel1]
        # qyel1 = (yes1==1) * qy[iel1] 
        # >>>>>>>> Here they are computed on the fly
        Tel1   = (yes1==1) * be[iel1]/ae[iel1]
        qxel1  = (yes1==1) * -ze[iel1,1]/mesh.Ω[iel1] 
        qyel1  = (yes1==1) * -ze[iel1,2]/mesh.Ω[iel1]
        @inbounds for ifac=1:mesh.nf_el
            # Face
            nodei = mesh.e2f[iel1,ifac]
            bcf   = mesh.bc[nodei]
            Γ     = mesh.Γ[iel1,ifac]
            ni_x  = mesh.n_x[iel1,ifac]
            ni_y  = mesh.n_y[iel1,ifac]
            τ     = mesh.τ[nodei]      
            # Assemble
            Tel1  += (yes1==1) * (bcf!=1) *  Γ*τ*Th[mesh.e2f[iel1, ifac]]/ae[iel1]
            qxel1 -= (yes1==1) * (bcf!=1) *  Γ*ni_x*Th[mesh.e2f[iel1, ifac]]/mesh.Ω[iel1]
            qyel1 -= (yes1==1) * (bcf!=1) *  Γ*ni_y*Th[mesh.e2f[iel1, ifac]]/mesh.Ω[iel1]
        end

        F1 = (bc!=1) * (yes1==1) * Γ1*(ni_x1*qxel1 + ni_y1*qyel1 + τ1*(Tel1 - Th[idof]) + Xi*ti)
        #######################################################

        # Element 2
        yes2  = mesh.f2e[idof,2]>0
        iel2  = (yes2==1) * mesh.f2e[idof,2]   + (yes2==0) * 1 # if the neigbouring element is out then set index to 1
        Γ2    = (yes2==1) * mesh.Γ_f[idof,2]  
        ni_x2 = (yes2==1) * mesh.n_x_f[idof,2] 
        ni_y2 = (yes2==1) * mesh.n_y_f[idof,2]
        τ2    = (yes2==1) * mesh.τ[idof]
        # >>>>>>>> Here, I assume Te, qx and qy are precomputed
        # Tel2  = (yes2==1) * Te[iel2]
        # qxel2 = (yes2==1) * qx[iel2]
        # qyel2 = (yes2==1) * qy[iel2]
        # >>>>>>>> Here they are computed on the fly
        Tel2   = (yes2==1) * be[iel2]/ae[iel2]
        qxel2  = (yes2==1) * -ze[iel2,1]/mesh.Ω[iel2]
        qyel2  = (yes2==1) * -ze[iel2,2]/mesh.Ω[iel2]
        @inbounds for ifac=1:mesh.nf_el
            # Face
            nodei = mesh.e2f[iel2,ifac]
            bcf   = mesh.bc[nodei]
            Γ     = mesh.Γ[iel2,ifac]
            ni_x  = mesh.n_x[iel2,ifac]
            ni_y  = mesh.n_y[iel2,ifac]
            τ     = mesh.τ[nodei]
            # Assemble
            Tel2  += (yes2==1) * (bcf!=1) *  Γ*τ*Th[mesh.e2f[iel2, ifac]]/ae[iel2]
            qxel2 -= (yes2==1) * (bcf!=1) *  Γ*ni_x*Th[mesh.e2f[iel2, ifac]]/mesh.Ω[iel2]
            qyel2 -= (yes2==1) * (bcf!=1) *  Γ*ni_y*Th[mesh.e2f[iel2, ifac]]/mesh.Ω[iel2]
        end

        # nitze  = ni_x*ze[iel,1] + ni_y*ze[iel,2]
        # fe_i   = (bci!=1) * -dAi * (mesh.ke[iel]/mesh.Ω[iel]*nitze - ti*Xi - 1.0/ae[iel]*be[iel]*taui)

        F2 = (bc!=1) * (yes2==1) * Γ2*(ni_x2*qxel2 + ni_y2*qyel2 + τ2*(Tel2 - Th[idof]) + Xi*ti)

        #######################################################
        
        # Contributions from the 2 elements
        F[idof] = F1 + F2
    end
    return F
end

#--------------------------------------------------------------------#

function ComputeElementValuesFaces(mesh, uh, ae, be, ze)
    ue          = zeros(mesh.nel);
    qx          = zeros(mesh.nel);
    qy          = zeros(mesh.nel);

    for idof=1:mesh.nf #

        # Identify BC faces
        bc = mesh.bc[idof]

        # Element 1
        yes1  = mesh.f2e[idof,1]>0
        iel1  = (yes1==1) * mesh.f2e[idof,1]  + (yes1==0) * 1 # if the neigbouring element is out then set index to 1
        Γ1    = (yes1==1) * mesh.Γ_f[idof,1]  
        ni_x1 = (yes1==1) * mesh.n_x_f[idof,1] 
        ni_y1 = (yes1==1) * mesh.n_y_f[idof,1] 
        τ1    = (yes1==1) * mesh.τ[idof]

        ue[iel1] += (yes1==1) * 1.0/mesh.nf_el*be[iel1]/ae[iel1]
        ue[iel1] += (bc!=1) * (yes1==1) * Γ1*τ1*uh[idof]/ae[iel1]

        qx[iel1] += (yes1==1) * 1.0/mesh.nf_el* -1.0/mesh.Ω[iel1]*ze[iel1,1]
        qx[iel1] -= (bc!=1) * (yes1==1) * 1.0/mesh.Ω[iel1]*Γ1*ni_x1*uh[idof]

        qy[iel1] += (yes1==1) * 1.0/mesh.nf_el* -1.0/mesh.Ω[iel1]*ze[iel1,2]
        qy[iel1] -= (bc!=1) * (yes1==1) * 1.0/mesh.Ω[iel1]*Γ1*ni_y1*uh[idof]

        # Element 2
        yes2  = mesh.f2e[idof,2]>0
        iel2  = (yes2==1) * mesh.f2e[idof,2]   + (yes2==0) * 1 # if the neigbouring element is out then set index to 1
        Γ2    = (yes2==1) * mesh.Γ_f[idof,2]   
        ni_x2 = (yes2==1) * mesh.n_x_f[idof,2] 
        ni_y2 = (yes2==1) * mesh.n_y_f[idof,2]
        τ2    = (yes2==1) * mesh.τ[idof]

        ue[iel2] += (yes2==1) * 1.0/mesh.nf_el*be[iel2]/ae[iel2]
        ue[iel2] += (bc!=1) * (yes2==1) * Γ2*τ2*uh[idof]/ae[iel2]
        
        qx[iel2] -= (yes2==1) * 1.0/mesh.nf_el* 1.0/mesh.Ω[iel2]*ze[iel2,1]
        qx[iel2] -= (bc!=1) * (yes2==1) * 1.0/mesh.Ω[iel2]*Γ2*ni_x2*uh[idof]

        qy[iel2] -= (yes2==1) * 1.0/mesh.nf_el* 1.0/mesh.Ω[iel2]*ze[iel2,2]
        qy[iel2] -= (bc!=1) * (yes2==1) * 1.0/mesh.Ω[iel2]*Γ2*ni_y2*uh[idof]

    end
    return ue, qx, qy
end

#--------------------------------------------------------------------#
