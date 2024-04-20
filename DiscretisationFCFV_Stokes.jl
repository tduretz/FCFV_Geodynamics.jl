# @turbo was removed
function ComputeFCFV(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu)

    α = zeros(mesh.nel)
    β = zeros(mesh.nel,2)
    Ζ = zeros(mesh.nel,2,2)

    # Assemble FCFV elements
    @inbounds for e=1:mesh.nel  

        β[e,1] += mesh.Ω[e]*sex[e]
        β[e,2] += mesh.Ω[e]*sey[e]
        
        for i=1:mesh.nf_el
            
            nodei = mesh.e2f[e,i]
            bc    = mesh.bc[nodei]
            Γi    = mesh.Γ[e,i]
            ni_x  = mesh.n_x[e,i]
            ni_y  = mesh.n_y[e,i]
            τi    = mesh.τ[nodei]  # Stabilisation parameter for the face

            # Assemble
            Ζ[e,1,1] += (bc==1) * Γi*ni_x*VxDir[nodei] # Dirichlet
            Ζ[e,1,2] += (bc==1) * Γi*ni_x*VyDir[nodei] # Dirichlet
            Ζ[e,2,1] += (bc==1) * Γi*ni_y*VxDir[nodei] # Dirichlet
            Ζ[e,2,2] += (bc==1) * Γi*ni_y*VyDir[nodei] # Dirichlet
            β[e,1]   += (bc==1) * Γi*τi*VxDir[nodei]   # Dirichlet
            β[e,2]   += (bc==1) * Γi*τi*VyDir[nodei]   # Dirichlet
            α[e]     +=           Γi*τi
            
        end
    end
    return α, β, Ζ
end

#--------------------------------------------------------------------#

function ComputeElementValues(mesh, Vxh, Vyh, Pe, α, β, Ζ, VxDir, VyDir)

    Vxe         = zeros(mesh.nel);
    Vye         = zeros(mesh.nel);
    Txxe        = zeros(mesh.nel);
    Tyye        = zeros(mesh.nel);
    Txye        = zeros(mesh.nel);

    @inbounds for e=1:mesh.nel
    
        η       =  mesh.ke[e]
        Ω       =  mesh.Ω[e]
        Vxe[e]  =  β[e,1]/α[e]
        Vye[e]  =  β[e,2]/α[e]
        Txxe[e] =  η/Ω*Ζ[e,1,1]
        Tyye[e] =  η/Ω*Ζ[e,2,2] 
        Txye[e] =  η/Ω*0.5*(Ζ[e,1,2] + Ζ[e,2,1])
        
        for i=1:mesh.nf_el
            
            # Face
            nodei = mesh.e2f[e,i]
            bc    = mesh.bc[nodei]
            Γi    = mesh.Γ[e,i]
            ni_x  = mesh.n_x[e,i]
            ni_y  = mesh.n_y[e,i]
            τi    = mesh.τ[nodei]  # Stabilisation parameter for the face

            # Assemble
            Vxe[e]  += (bc!=1) *  Γi*τi*Vxh[nodei]/α[e]
            Vye[e]  += (bc!=1) *  Γi*τi*Vyh[nodei]/α[e]
            Txxe[e] += (bc!=1) *  η/Ω*Γi*ni_x*Vxh[nodei]
            Tyye[e] += (bc!=1) *  η/Ω*Γi*ni_y*Vyh[nodei]
            Txye[e] += (bc!=1) *  η*0.5*( 1.0/Ω*Γi*( ni_x*Vyh[nodei] + ni_y*Vxh[nodei] ) )
         end
        Txxe[e] *= 2.0
        Tyye[e] *= 2.0
        Txye[e] *= 2.0
    end
    return Vxe, Vye, Txxe, Tyye, Txye
end

#--------------------------------------------------------------------#

function ElementAssemblyLoop(mesh, α, β, Ζ, VxDir, VyDir, σxxNeu, σyyNeu, σxyNeu, σyxNeu, gbar, new) 

    # Assemble element matrices and rhs
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

    @inbounds for e=1:mesh.nel 

        # Element properties
        Ωe = mesh.Ω[e]
        ηe = mesh.ke[e]

        for i=1:mesh.nf_el 

            ni_x, ni_y = mesh.n_x[e,i], mesh.n_y[e,i]
            nodei = mesh.e2f[e,i]
            bci   = mesh.bc[nodei]
            ȷ     = 0.0 + (bci==3)*1.0 # indicates interface
            Γi    = mesh.Γ[e,i]
            τi    = mesh.τ[nodei]  

            # if ȷ==1
            # ηn = mesh.ke[mesh.e2e[e,i]]
            #     if ηe==1.0
            #         ηe = 2.0/(1.0 + 1.0/10.0)
            #         ηe = (1.0 + 10.0)/2
            #     else
            #         ηe = (1.0 + 10.0)/2
            #         # ηe = sqrt(1*10.0)
            #     end
            #     # println(ηe)
            # end
                
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

                # Element matrix 
                Kuuv[j   , i   , e] = on * -Γi * (α[e]^-1*τi*τj*Γj - ηe*Ωe^-1*Γj*(ninj + new*ȷ*ni_x*nj_x) - τi*δ) # u1u1
                Kuuv[j+nf, i   , e] = on * -Γi * (                 - ηe*Ωe^-1*Γj*(       new*ȷ*ni_y*nj_x)       ) # u1u2
                Kuuv[j   , i+nf, e] = on * -Γi * (                 - ηe*Ωe^-1*Γj*(       new*ȷ*ni_x*nj_y)       ) # u2u1
                Kuuv[j+nf, i+nf, e] = on * -Γi * (α[e]^-1*τi*τj*Γj - ηe*Ωe^-1*Γj*(ninj + new*ȷ*ni_y*nj_y) - τi*δ) # u2u2

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
            feix  = (bci!=1) * -Γi * (-α[e]^-1*τi*β[e,1] + ηe*Ωe^-1*niΖ_x - tix*Xi - (1-new)*ȷ*gbar[e,i,1])
            feiy  = (bci!=1) * -Γi * (-α[e]^-1*τi*β[e,2] + ηe*Ωe^-1*niΖ_y - tiy*Xi - (1-new)*ȷ*gbar[e,i,2])
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

function CreateTripletsSparse(mesh, Kuu_v, fu_v, Kup_v)
    # ACHTUNG: This function is deprecated since it gives wrong xy connectivity
    # Create triplets and assemble sparse matrix for Kuu
    e2fu = mesh.e2f
    e2fv = mesh.e2f .+ mesh.nf 
    e2f  = hcat(e2fu, e2fv)
    idof = 1:mesh.nf_el*2  
    ii   = repeat(idof, 1, length(idof))'
    ij   = repeat(idof, 1, length(idof))
    Ki   = e2f[:,ii]
    Kif  = e2f[:,ii[1,:]] # for RHS
    Kj   = e2f[:,ij]
    @time Kuu  = sparse(Ki[:], Kj[:], Kuu_v[:], mesh.nf*2, mesh.nf*2)
    
    # file = matopen(string(@__DIR__,"/results/matrix_uu.mat"), "w" )
    # write(file, "Ki",       Ki[:] )
    # write(file, "Kj",    Kj[:] )
    # write(file, "Kuu",  Kuu_v[:] )
    # write(file, "nrow",  mesh.nf*2 )
    # write(file, "ncol",  mesh.nf*2 )
    # close(file)
    @time fu   = sparse(Kif[:], ones(size(Kif[:])), fu_v[:], mesh.nf*2, 1)
    fu   = Array(fu)
    droptol!(Kuu, 1e-6)
    # Create triplets and assemble sparse matrix fo Kup
    idof = 1:mesh.nf_el*2  
    ii   = repeat(idof, 1, mesh.nel)'
    ij   = repeat(1:mesh.nel, 1, length(idof))
    Ki   = e2f
    Kj   = ij
    @time Kup  = sparse(Ki[:], Kj[:], Kup_v[:], mesh.nf*2, mesh.nel  )
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

@doc """
Computes residual of local and global equations discretised with FCFV.
""" ComputeResidualsFCFV_Stokes_o1

function ComputeResidualsFCFV_Stokes_o1(Vxh, Vyh, Pe, mesh, ae, be, ze, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, Formulation)

    # reconstruct element value and flux
    nel = mesh.nel
    ue  = zeros(nel,2)
    Le  = zeros(nel,2,2)

    for e=1:nel    
        nfac      = mesh.nf_el
        vole      = mesh.Ω[e]
        ue[e,:]   = be[e,:]/ae[e]
        Le[e,:,:] = -1.0/vole * ze[e,:,:]
        τi        = mesh.τ[e]
        for i=1:nfac
            nodei = mesh.e2f[e,i]
            if mesh.bc[nodei] != 1
                Γi        = mesh.Γ[e,i]
                n         = [mesh.n_x[e,i]; mesh.n_y[e,i]]
                u         = [Vxh[nodei]; Vyh[nodei]]
                ue[e,:]   = ue[e,:]   .+ Γi*τi*u'[:]/ae[e]
                if Formulation == :Gradient
                    Le[e,:,:] = Le[e,:,:] .- 1.0/vole * Γi * (n*u')
                elseif Formulation == :SymmetricGradient
                    Le[e,:,:] = Le[e,:,:] .- 1.0/vole * Γi * (n*u' .+ u*n')
                end
            end
        end
    end

    # Check out the 3 residuals: 3 for local equations and 2 for the global
    F_glob1  = zeros(nel,mesh.nf_el,2);
    F_glob2  = zeros(nel);
    F_eq1    = zeros(nel,2,2);
    F_eq2    = zeros(nel,2);
    F_eq3    = zeros(nel,1);
    D        = zeros(2,2)

    for e=1:nel
        nfac         = mesh.nf_el
        vole         = mesh.Ω[e]
        η            = mesh.ke[e]
        D           .= [η  η; η η]
        F_eq1[e,:,:] = vole*Le[e,:,:]
        F_eq2[e,1]   = -sex[e]*vole
        F_eq2[e,2]   = -sey[e]*vole
        F_eq3[e]     = 0.0
        for i=1:nfac
            nodei  = mesh.e2f[e,i]
            dAi    = mesh.Γ[e,i]
            taui   = mesh.τ[nodei]  
            n      = [mesh.n_x[e,i]; mesh.n_y[e,i]]
            bci    = mesh.bc[nodei]
            Xi     = 0.0 + (bci== 2)*1.0
            Ji     = 0.0 + (bci==-1)*1.0
            tix    = n[1]*SxxNeu[nodei] + n[2]*SxyNeu[nodei]
            tiy    = n[1]*SyxNeu[nodei] + n[2]*SyyNeu[nodei] 
            ti     = [tix; tiy]
            if mesh.bc[nodei] == 1
                u = [VxDir[nodei]; VyDir[nodei]]
            else
                u = [Vxh[nodei]; Vyh[nodei]]
            end
            # Global residual 1 (momentum) 
            if Formulation == :Gradient
                F_glob1[e,i,:] .+=  dAi .* ( (n'*(D.*Le[e,:,:])) .+ Pe[e]*n' .+ taui*ue[e,:]' .- taui*u' .+ ti'*Xi .+ Ji*(n'*(D.*Le[e,:,:])') )'
            elseif Formulation == :SymmetricGradient
                F_glob1[e,i,:] .+=  dAi .* ( (n'*(D.*Le[e,:,:])) .+ Pe[e]*n' .+ taui*ue[e,:]' .- taui*u' .+ ti'*Xi  )'
            end
            F_eq2[e,:]       = F_eq2[e,:] + dAi*taui*ue[e,:]
            # Global residual 2 (continuity)
            F_glob2[e]   = F_glob2[e] .+ dAi * u'*n
            if Formulation == :Gradient
                F_eq1[e,:,:] = F_eq1[e,:,:] .+ dAi * n * u'
            elseif Formulation == :SymmetricGradient
                F_eq1[e,:,:] = F_eq1[e,:,:] .+ dAi * (n*u' .+ u*n')
            end
            F_eq2[e,:] = F_eq2[e,:] .- dAi*taui*u[:]
            F_eq3[e,:] = F_eq3[e,:] .+ dAi*u'*n
        end
    end

    # Loop through elements and add global residual contribution of eac
    # individual faces to global nodal residual
    nnod = mesh.nf
    F_nodes_x = zeros(nnod,1);
    F_nodes_y = zeros(nnod,1);
    for e=1:nel
        nfac = mesh.nf_el
        for i=1:nfac
            nodei = mesh.e2f[e,i]
            if mesh.bc[nodei] != 1
                F_nodes_x[nodei] += F_glob1[e,i,1] 
                F_nodes_y[nodei] += F_glob1[e,i,2] 
            end
        end
    end
    @printf("Residual of local  equation 20a: %2.2e\n", norm(F_eq1)/sqrt(length(F_eq1)))
    @printf("Residual of local  equation 20b: %2.2e\n", norm(F_eq2)/sqrt(length(F_eq2)))
    @printf("Residual of local  equation 20c: %2.2e\n", norm(F_eq3)/sqrt(length(F_eq3)))
    @printf("Residual of global equation 19a: %2.2e\n", norm(F_nodes_x)/sqrt(length(F_nodes_x)))
    @printf("Residual of global equation 19a: %2.2e\n", norm(F_nodes_y)/sqrt(length(F_nodes_y)))
    @printf("Residual of global equation 19b: %2.2e\n", norm(F_glob2)/sqrt(length(F_glob2)))

    return F_nodes_x, F_nodes_y
end

#--------------------------------------------------------------------#
