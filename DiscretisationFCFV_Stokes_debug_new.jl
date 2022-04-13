# @turbo was removed
function ComputeFCFV(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, tau)
    ae = zeros(mesh.nel)
    be = zeros(mesh.nel,2)
    ze = zeros(mesh.nel,2,2)

    # Assemble FCFV elements
    for e=1:mesh.nel  

        be[e,1] += mesh.vole[e]*sex[e]
        be[e,2] += mesh.vole[e]*sey[e]
        
        for i=1:mesh.nf_el
            
            nodei = mesh.e2f[e,i]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[e,i]
            ni_x  = mesh.n_x[e,i]
            ni_y  = mesh.n_y[e,i]
            taui  = StabParam(tau, dAi, mesh.vole[e], mesh.type, mesh.ke[e])                              # Stabilisation parameter for the face

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
    return ae, be, ze
end

#--------------------------------------------------------------------#

function ComputeElementValues(mesh, Vxh, Vyh, Pe, ae, be, ze, VxDir, VyDir, tau)
    Vxe         = zeros(mesh.nel);
    Vye         = zeros(mesh.nel);
    Txxe        = zeros(mesh.nel);
    Tyye        = zeros(mesh.nel);
    Txye        = zeros(mesh.nel);

    for e=1:mesh.nel
    
        Vxe[e]  =  be[e,1]/ae[e]
        Vye[e]  =  be[e,2]/ae[e]
        Txxe[e] =  mesh.ke[e]/mesh.vole[e]*ze[e,1,1]
        Tyye[e] =  mesh.ke[e]/mesh.vole[e]*ze[e,2,2] 
        Txye[e] =  mesh.ke[e]/mesh.vole[e]*0.5*(ze[e,1,2]+ze[e,2,1])
        
        for i=1:mesh.nf_el
            
            # Face
            nodei = mesh.e2f[e,i]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[e,i]
            ni_x  = mesh.n_x[e,i]
            ni_y  = mesh.n_y[e,i]
            taui  = StabParam(tau, dAi, mesh.vole[e], mesh.type, mesh.ke[e])      # Stabilisation parameter for the face

            # Assemble
            Vxe[e]  += (bc!=1) *  dAi*taui*Vxh[nodei]/ae[e]
            Vye[e]  += (bc!=1) *  dAi*taui*Vyh[nodei]/ae[e]
            Txxe[e] += (bc!=1) *  mesh.ke[e]/mesh.vole[e]*dAi*ni_x*Vxh[nodei]
            Tyye[e] += (bc!=1) *  mesh.ke[e]/mesh.vole[e]*dAi*ni_y*Vyh[nodei]
            Txye[e] += (bc!=1) *  mesh.ke[e]*0.5*( 1.0/mesh.vole[e]*dAi*( ni_x*Vyh[nodei] + ni_y*Vxh[nodei] ) )
         end
        Txxe[e] *= 2.0
        Tyye[e] *= 2.0
        Txye[e] *= 2.0
    end
    return Vxe, Vye, Txxe, Tyye, Txye
end

#--------------------------------------------------------------------#

function ElementAssemblyLoop(mesh, α, β, Ζ, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, tau) 
    # Assemble element matrices and rhs
    Ki  = zeros(mesh.nel, 2*mesh.nf_el, 2*mesh.nf_el)
    Kj  = zeros(mesh.nel, 2*mesh.nf_el, 2*mesh.nf_el)
    Kuu = zeros(mesh.nel, 2*mesh.nf_el, 2*mesh.nf_el)
    fu  = zeros(mesh.nel, 2*mesh.nf_el)
    Kup = zeros(mesh.nel, 2*mesh.nf_el);
    fp  = zeros(mesh.nel)
    nf  = mesh.nf_el

    Kuug   = spzeros(2*mesh.nf, 2*mesh.nf)

    new  = 1

    for e=1:mesh.nel 

        Ωe = mesh.vole[e]

        HasInterface=0
        Kij = 0
        for i=1:mesh.nf_el 

            ni_x, ni_y = mesh.n_x[e,i], mesh.n_y[e,i]
            nodei = mesh.e2f[e,i]
            bci   = mesh.bc[nodei]
            ȷ     = 0.0 + (bci==3)*1.0 # indicates interface
            if (ȷ==1) HasInterface = 1 end
            # if (ȷ==1) @printf("i = %d\n", i) end
            Γi    = mesh.dA[e,i]
            τi    = StabParam(tau, Γi, Ωe, mesh.type, mesh.ke[e])  
            νe    = mesh.ke[e]
                
            for j=1:mesh.nf_el

                nj_x, nj_y  = mesh.n_x[e,j], mesh.n_y[e,j]
                nodej = mesh.e2f[e,j]
                bcj   = mesh.bc[nodej]   
                Γj    = mesh.dA[e,j]
                τj    = StabParam(tau, Γj, Ωe, mesh.type, νe)   
                δ     = 0.0 + (i==j)*1.0    # Delta operator
                on    = (bci!=1) & (bcj!=1) # Activate nodal connection if not Dirichlet!
                        
                # Element matrix components
                ninj = ni_x*nj_x + ni_y*nj_y

                I = [[1,0] [0,1]]
                nI = [ni_x, ni_y ]
                nJ = [nj_x, nj_y ]

                if ȷ==1
                    aux1 = Γj*νe/Ωe*(dot(nJ,nI)*I .+ new*nJ*nI')
                else
                    aux1 = Γj*νe/Ωe*(dot(nJ,nI)*I)
                    # display(dot(nJ,nI))
                end
                
                aux2 = Γj*τi*τj/α[e]*I;
                aux3 = τi*δ*I;

                Kij = Γi.*( .-aux1 .+ aux2 .- aux3 )
            
                # Element matrix 
                Kuu[e, i,    j   ] = on * -Γi * (α[e]^-1*τi*τj*Γj - νe*Ωe^-1*Γj*(ninj + new*ȷ*ni_x*nj_x) - τi*δ) # u1u1
                Kuu[e, i,    j+nf] = on * -Γi * (                 - νe*Ωe^-1*Γj*(       new*ȷ*ni_y*nj_x)       ) # u1u2
                Kuu[e, i+nf, j   ] = on * -Γi * (                 - νe*Ωe^-1*Γj*(       new*ȷ*ni_x*nj_y)       ) # u2u1
                Kuu[e, i+nf, j+nf] = on * -Γi * (α[e]^-1*τi*τj*Γj - νe*Ωe^-1*Γj*(ninj + new*ȷ*ni_y*nj_y) - τi*δ) # u2u2


                # Kuu[e, i,    j   ] = on * Kij[1,1]#Γi * (α[e]^-1*τi*τj*Γj - νe*Ωe^-1*Γj*(ninj + new*ȷ*ni_x*nj_x) - τi*δ) # u1u1
                # Kuu[e, i,    j+nf] = on * Kij[1,2]#Γi * (                 - νe*Ωe^-1*Γj*(       new*ȷ*ni_y*nj_x)       ) # u1u2
                # Kuu[e, i+nf, j   ] = on * Kij[2,1]#Γi * (                 - νe*Ωe^-1*Γj*(       new*ȷ*ni_x*nj_y)       ) # u2u1
                # Kuu[e, i+nf, j+nf] = on * Kij[2,2]#Γi * (α[e]^-1*τi*τj*Γj - νe*Ωe^-1*Γj*(ninj + new*ȷ*ni_y*nj_y) - τi*δ) # u2u2

                Ki[e, i,    j   ]  = nodei
                Kj[e, i,    j   ]  = nodej
                Ki[e, i,    j+nf]  = nodei
                Kj[e, i,    j+nf]  = nodej+mesh.nf
                Ki[e, i+nf,    j   ]  = nodei+mesh.nf
                Kj[e, i+nf,    j   ]  = nodej
                Ki[e, i+nf,    j+nf]  = nodei+mesh.nf
                Kj[e, i+nf,    j+nf]  = nodej+mesh.nf

            #     if ȷ==1
            #         display(Kij)
            #         println(Γi * (α[e]^-1*τi*τj*Γj - νe*Ωe^-1*Γj*(ninj + new*ȷ*ni_x*nj_x) - τi*δ))
            #         println(Γi * (                 - νe*Ωe^-1*Γj*(       new*ȷ*ni_y*nj_x)       ))
            #         println(Γi * (                 - νe*Ωe^-1*Γj*(       new*ȷ*ni_x*nj_y)       ))
            #         println(Γi * (α[e]^-1*τi*τj*Γj - νe*Ωe^-1*Γj*(ninj + new*ȷ*ni_y*nj_y) - τi*δ))
            #    end

                
                Kuug[nodei, nodej] += on * Kij[1,1]
                Kuug[nodei, nodej+mesh.nf] += on * Kij[1,2]
                Kuug[nodei+mesh.nf, nodej] += on * Kij[2,1]
                Kuug[nodei+mesh.nf, nodej+mesh.nf] += on * Kij[2,2]

            end 
            # RHS
            Xi    = 0.0 + (bci==2)*1.0
            tix   = ni_x*SxxNeu[nodei] + ni_y*SxyNeu[nodei]
            tiy   = ni_x*SyxNeu[nodei] + ni_y*SyyNeu[nodei]   
            niΖ_x = ni_x*(Ζ[e,1,1] +  new*ȷ*Ζ[e,1,1]) + ni_y*(Ζ[e,2,1] + new*ȷ*Ζ[e,1,2]) 
            niΖ_y = ni_x*(Ζ[e,1,2] +  new*ȷ*Ζ[e,2,1]) + ni_y*(Ζ[e,2,2] + new*ȷ*Ζ[e,2,2])
            feix  = (bci!=1) * -Γi * (-α[e]^-1*τi*β[e,1] + νe*Ωe^-1*niΖ_x - tix*Xi - (1-new)*ȷ*gbar[e,i,1])
            feiy  = (bci!=1) * -Γi * (-α[e]^-1*τi*β[e,2] + νe*Ωe^-1*niΖ_y - tiy*Xi - (1-new)*ȷ*gbar[e,i,2])
            # up block
            Kup[e, i   ]       -= (bci!=1) * Γi*ni_x
            Kup[e, i+nf]       -= (bci!=1) * Γi*ni_y
            # Dirichlet nodes - uu block
            Kuu[e, i   , i   ] += (bci==1) * 1e0
            Kuu[e, i+nf, i+nf] += (bci==1) * 1e0
            fu[e, i   ]        += (bci!=1) * feix + (bci==1) * VxDir[nodei] * 1e0
            fu[e, i+nf]        += (bci!=1) * feiy + (bci==1) * VyDir[nodei] * 1e0
            # Dirichlet nodes - pressure RHS
            fp[e]              -= (bci==1) * Γi*(VxDir[nodei]*ni_x + VyDir[nodei]*ni_y)

            if bci==1
                Kuug[nodei,nodei] = 1.0
                Kuug[nodei+mesh.nf,nodei+mesh.nf] = 1.0
            end
        end
    end
    Kuua  = sparse(Ki[:], Kj[:], Kuu[:], mesh.nf*2, mesh.nf*2)
    return Kuu, fu, Kup, fp, Kuug, Kuua
end

#--------------------------------------------------------------------#

function CreateTripletsSparse(mesh, Kuu_v, fu_v, Kup_v)
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
    
    file = matopen(string(@__DIR__,"/results/matrix_uu.mat"), "w" )
    write(file, "Ki",       Ki[:] )
    write(file, "Kj",    Kj[:] )
    write(file, "Kuu",  Kuu_v[:] )
    write(file, "nrow",  mesh.nf*2 )
    write(file, "ncol",  mesh.nf*2 )
    close(file)
    @time fu   = sparse(Kif[:], ones(size(Kif[:])), fu_v[:], mesh.nf*2, 1)
    u   = Array(fu)
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