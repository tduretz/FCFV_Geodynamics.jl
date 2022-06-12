const USE_GPU    = false  # not supported yet 
const USE_DIRECT = false
const USE_NODAL  = false
const USE_MAKIE  = false

using Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT

include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../DiscretisationFCFV.jl")
include("IntegrationPoints.jl")

#----------------------------------------------------------#

function StabParam(τ, Γ, Ω, mesh_type, ν) 
    return 0. # Stabilisation is only needed for FCFV
end

#----------------------------------------------------------#

function ResidualPoissonNodalFEM!( F, T, mesh, K_all, b )
    # Residual
    @inbounds for in = 1:mesh.nv
        F[in] = 0.0
        if mesh.bcn[in]==0
            for ii=1:length(mesh.n2e[in])
                e       = mesh.n2e[in][ii]
                nodes   = mesh.e2n[e,:]
                T_ele   = T[nodes]  
                K       = K_all[e,:,:] 
                f       = K*T_ele .- b[e]
                inode   = mesh.n2e_loc[in][ii]
                F[in]  -= f[inode]
            end
        end
    end
    return nothing
end

#----------------------------------------------------------#

function ResidualPoissonElementalFEM!( F, T, mesh, K_all, b )
    # Residual
    F .= 0.0
    @inbounds for e = 1:mesh.nel
        nodes      = mesh.e2n[e,:]
        T_ele      = T[nodes]
        K_ele      = K_all[e,:,:]
        f_ele      = K_ele*T_ele .- b[e,:]
        F[nodes] .-= f_ele[:] # This should be atomic
    end
    F[mesh.bcn.==1] .= 0.0
    return nothing
end

#----------------------------------------------------------#

function ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX ) 

    nnel         = mesh.nnel
    nip          = length(ipw)
    K_all        = zeros(mesh.nel, nnel, nnel) 
    b            = zeros(mesh.nel, nnel)
    K_ele        = zeros(nnel, nnel)
    b_ele        = zeros(nnel)

    # Element loop
    @inbounds for e = 1:mesh.nel
        nodes   = mesh.e2n[e,:]
        x       = [mesh.xv[nodes] mesh.yv[nodes]]    
        ke      = mesh.ke[e]
        J       = zeros(2,2)
        invJ    = zeros(2,2)
        K_ele  .= 0.0
        b_ele  .= 0.0
        # Integration loop
        for ip=1:nip
            dNdXi     = dNdX[ip,:,:]
            J        .= x'*dNdXi
            detJ      = J[1,1]*J[2,2] - J[1,2]*J[2,1]
            invJ[1,1] = +J[2,2] / detJ
            invJ[1,2] = -J[2,1] / detJ
            invJ[2,1] = -J[1,2] / detJ
            invJ[2,2] = +J[1,1] / detJ
            dNdx      = dNdXi*invJ
            K_ele   .+= ipw[ip] .* detJ .* ke .* (dNdx*dNdx')
            b_ele   .+= ipw[ip] .* detJ .* se[e] .* N[ip,:] 
        end
        K_all[e,:,:] .= K_ele
        b[e,:]       .= b_ele
    end
    return K_all, b
end 

#----------------------------------------------------------#

function DirectSolveFEM!(T, mesh, K_all, b)

    ndof = length(T)
    K    = sparse(Int64[], Int64[], Float64[], ndof, ndof)
    rhs  = zeros(ndof)

    # Assembly of global sparse matrix
    @inbounds for e=1:mesh.nel
        nodes = mesh.e2n[e,:]
        for j=1:3
            if mesh.bcn[nodes[j]] == 0
                # If not Dirichlet, add connection
                for i=1:3  
                    if mesh.bcn[nodes[i]] == 0 # Important to keep matrix symmetric
                        K[nodes[j], nodes[i]] += K_all[e,j,i]
                    else
                        rhs[nodes[j]] -= K_all[e,j,i]*T[nodes[i]]
                    end
                end
                rhs[nodes[j]] += b[e,j]
            else
                # Deal with Dirichlet: set one on diagonal and value and RHS
                rhs[nodes[j]]         = T[nodes[j]]
                K[nodes[j], nodes[j]] = 1.0
            end
        end 
    end
    # Solve using CHOLDMOD
    T .= cholesky(K)\rhs
    return nothing
end

#----------------------------------------------------------#

function main( n, θ, Δτ )

    println("\n******** FEM POISSON ********")
    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = Int16(n*20), Int16(n*20)
    R          = 0.5
    inclusion  = 0

    # Element data
    mesh_type  = "UnstructTriangles" 
    nip       = 3
    nnel      = 3
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)
  
    # Generate mesh
    if mesh_type=="Quadrangles" # not supported for FEM
        τr   = 1
        # mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R )
    elseif mesh_type=="UnstructTriangles"  
        τr   = 1
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R; nnel ) 
    end
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nv)

    T  = zeros(mesh.nv)     # Solution on nodes 
    se = zeros(mesh.nel)    # Source term on elements

    # Intial guess
    alp = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;
    for in = 1:mesh.nv
        if mesh.bcn[in]==1
            x     = mesh.xv[in]
            y     = mesh.yv[in]
            T[in] = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        end
    end

    # Evaluate T analytic on barycentres
    for e=1:mesh.nel
        x      = mesh.xc[e]
        y      = mesh.yc[e]
        Te     = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        se[e] = Te*(-a*alp*cos(a*x + c*y) + b*bet*sin(b*x + d*y))*(a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y)) + Te*(a^2*alp*sin(a*x + c*y) + b^2*bet*cos(b*x + d*y)) + Te*(-alp*c*cos(a*x + c*y) + bet*d*sin(b*x + d*y))*(alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y)) + Te*(alp*c^2*sin(a*x + c*y) + bet*d^2*cos(b*x + d*y))
    end

    #-----------------------------------------------------------------#
    K_all, b = ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX )
    #-----------------------------------------------------------------#
    if USE_DIRECT
        @time DirectSolveFEM!(T, mesh, K_all, b)
    else
        nout    = 50#1e1
        iterMax = 2e3#5e4
        ϵ_PT    = 1e-7
        # θ       = 0.11428 *1.7
        # Δτ      = 0.28 /1.2
        # println(minimum(mesh.Γ))
        # println(maximum(mesh.Γ))
        # println(minimum(mesh.Ω))
        # println(maximum(mesh.Ω))
        # println("Δτ = ", Δτ)
        # Ωe = maximum(mesh.Ω)
        # Δx = minimum(mesh.Γ)
        # D  = 1.0
        # println("Δτ1 = ", Δx^2/(1.1*D) * 1.0/Ωe *2/3)
        ΔTΔτ  = zeros(mesh.nv) # Residual
        ΔTΔτ0 = zeros(mesh.nv)
        # PT loop
        local iter = 0
        success = 0
        @time while (iter<iterMax)
            iter +=1
            ΔTΔτ0 .= ΔTΔτ 
            if USE_NODAL
                ResidualPoissonNodalFEM!( ΔTΔτ, T, mesh, K_all, b )
            else
                ResidualPoissonElementalFEM!( ΔTΔτ, T, mesh, K_all, b )
            end
            ΔTΔτ  .= (1.0 - θ).*ΔTΔτ0 .+ ΔTΔτ 
            T    .+= Δτ .* ΔTΔτ
            if iter % nout == 0 || iter==1
                err = norm(ΔTΔτ)/sqrt(length(ΔTΔτ))
                @printf("PT Iter. %05d --- Norm of matrix-free residual: %3.3e\n", iter, err)
                if err < ϵ_PT
                    print("PT solve converged in")
                    success = true
                    break
                elseif err>1e4
                    success = false
                    println("exploding !")
                    break
                elseif isnan(err)
                    success = false
                    println("NaN !")
                    break
                end
            end
        end
    end

    #-----------------------------------------------------------------#

    Te = zeros(mesh.nel)
    for e=1:mesh.nel
        for in=1:mesh.nnel
            Te[e] += 1.0/3.0 * T[mesh.e2n[e,in]]
        end
    end

    if USE_MAKIE
        PlotMakie(mesh, Te, xmin, xmax, ymin, ymax)
    end
   
end

# main(1, 0.20398980000000003,        0.23333333333333336) # 150
main(2, 0.20398980000000003*0.61,   0.23333333333333336) # 250
# main(4, 0.20398980000000003*0.61/2, 0.23333333333333336*0.96) # 500
# main(8, 0.20398980000000003*0.61/4 * 0.98, 0.23333333333333336*0.88) # 1000

