const USE_GPU      = false  # Not supported yet 
const USE_DIRECT   = true   # Sparse matrix assembly + direct solver
const USE_NODAL    = false  # Nodal evaluation of residual
const USE_PARALLEL = false  # Parallel residual evaluation
const USE_MAKIE    = true   # Visualisation 
import Plots

using Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT#, StaticArrays
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid
import Statistics: mean
using MAT

include("FunctionsFEM.jl")
include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
# include("../EvalAnalDani.jl")
include("IntegrationPoints.jl")

#----------------------------------------------------------#

function ComputeStressFEM!( εxx, εyy, εxy, τxx, τyy, τxy, Vx, Vy, mesh, ipx, ipw, N, dNdX ) 
    ndof         = 2*mesh.nnel
    nnel         = mesh.nnel
    npel         = mesh.npel
    nip          = length(ipw)
    m            =  [ 1.0; 1.0; 0.0]
    Dev          =  [ 4/3 -2/3  0.0;
                     -2/3  4/3  0.0;
                      0.0  0.0  1.0]
    P  = ones(npel,npel)
    Pb = ones(npel)
    Np0, dNdXp   = ShapeFunctions(ipx, nip, 3)
    # Element loop
    @inbounds for e = 1:mesh.nel
        nodes   = mesh.e2n[e,:]
        x       = [mesh.xn[nodes] mesh.yn[nodes]]  
        ke      = mesh.ke[e]
        J       = zeros(2,2)
        invJ    = zeros(2,2)
        V_ele   = zeros(nnel,2)
    
        if npel==3 && nnel!=4 P[2:3,:] .= (x[1:3,:])' end

        # Integration loop
        for ip=1:nip
            dNdXi     = dNdX[ip,:,:]
            J        .= x'*dNdXi
            detJ      = J[1,1]*J[2,2] - J[1,2]*J[2,1]
            invJ[1,1] = +J[2,2] / detJ
            invJ[1,2] = -J[1,2] / detJ
            invJ[2,1] = -J[2,1] / detJ
            invJ[2,2] = +J[1,1] / detJ
            dNdx      = dNdXi*invJ
            V_ele[:,1] = Vx[nodes]
            V_ele[:,2] = Vy[nodes]
            εxxt      = dNdx[:,1]' * V_ele[:,1]
            εyyt      = dNdx[:,2]' * V_ele[:,2]
            εxyt      = 1//2*(dNdx[:,1]' * V_ele[:,2] + dNdx[:,2]' * V_ele[:,1])
            divV      = εxxt + εyyt
            εxx[e,ip] = εxxt - 1//3*divV
            εyy[e,ip] = εyyt - 1//3*divV
            εxy[e,ip] = εxyt
            τxx[e,ip] = 2*ke*εxx[e,ip]
            τyy[e,ip] = 2*ke*εyy[e,ip]
            τxy[e,ip] = 2*ke*εxy[e,ip]
            # εxx[e,ip] = dNdx
            # B[1:2:end,1] .= dNdx[:,1]
            # B[2:2:end,2] .= dNdx[:,2]
            # B[1:2:end,3] .= dNdx[:,2]
            # B[2:2:end,3] .= dNdx[:,1]
            # Bvol          = dNdx'
            # K_ele .+= ipw[ip] .* detJ .* ke  .* (B*Dev*B')
            # if npel==3 && nnel!=4 
            #     Np       = N[ip,:,1]
            #     Pb[2:3] .= x'*Np 
            #     Pi       = P\Pb
            #     Q_ele   .-= ipw[ip] .* detJ .* (Bvol[:]*Pi') 
            #     # M_ele   .+= ipw[ip] .* detJ .* Pi*Pi' # mass matrix P, not needed for incompressible
            # else
            #     if npel==1 Np = 1.0        end
            #     if npel==3 Np = Np0[ip,:,:] end
            #     Q_ele   .-= ipw[ip] .* detJ .* (B*m*Np') 
            # end
            # b_ele[1:2:end] .+= ipw[ip] .* detJ .* se[e,1] .* N[ip,:] 
            # b_ele[2:2:end] .+= ipw[ip] .* detJ .* se[e,2] .* N[ip,:]
        end
    end
    return nothing
end 

#----------------------------------------------------------#

function EvalSolSevilla2018( x, y )
    p   =  x*(1-x)
    Vx  =  x^2*(1 - x)^2*(4*y^3 - 6*y^2 + 2*y)
    Vy  = -y^2*(1 - y)^2*(4*x^3 - 6*x^2 + 2*x)
    Sxx = -8*p*y*(x - 1)*(2*y^2 - 3*y + 1) - p + 8*x^2*y*(x - 1)*(2*y^2 - 3*y + 1)
    Syy = -p - 8*x*y^2*(y - 1)*(2*x^2 - 3*x + 1) - 8*x*y*(y - 1)^2*(2*x^2 - 3*x + 1)
    Sxy = p^2*(12*y^2 - 12*y + 2) + y^2*(y - 1)^2*(-12.0*x^2 + 12.0*x - 2.0)
    sx  = -p^2*(24*y - 12) - 4*x^2*(4*y^3 - 6*y^2 + 2*y) - 8*x*(2*x - 2)*(4*y^3 - 6*y^2 + 2*y) - 2*x + 1.0*y^2*(2*y - 2)*(12*x^2 - 12*x + 2) + 2.0*y*(1 - y)^2*(12*x^2 - 12*x + 2) - 4*(1 - x)^2*(4*y^3 - 6*y^2 + 2*y) + 1
    sy  = -2*p*(1 - x)*(12*y^2 - 12*y + 2) - x^2*(2*x - 2)*(12*y^2 - 12*y + 2) + 1.0*y^2*(1 - y)^2*(24*x - 12) + 4*y^2*(4*x^3 - 6*x^2 + 2*x) + 8*y*(2*y - 2)*(4*x^3 - 6*x^2 + 2*x) + 4*(1 - y)^2*(4*x^3 - 6*x^2 + 2*x)
    return Vx, Vy, p, Sxx, Syy, Sxy, sx, sy
end

#----------------------------------------------------------#

function main( n, nnel, npel, nip, θ, ΔτV, ΔτP )

    println("\n******** FEM STOKES ********")
    # Create sides of mesh
    xmin, xmax = -.0, 1.0
    ymin, ymax = -.0, 1.0
    nx, ny     = Int16(n*30), Int16(n*30)
    R          = 1.0
    inclusion  = 0
    εBG        = 1.0
    η          = [1.0 5.0]  
    
    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)
  
    # Generate mesh
    mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R; nnel, npel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    println("Number of p nodes:  ", mesh.np)

    Vx  = zeros(mesh.nn)       # Solution on nodes 
    Vy  = zeros(mesh.nn)       # Solution on nodes 
    P   = zeros(mesh.np)       # Solution on either elements or nodes 
    se  = zeros(mesh.nel,2)    # Source on elements 

    Pa   = zeros(mesh.nel)      # Solution on elements 
    Sxxa = zeros(mesh.nel)      # Solution on elements 

    # Intial guess
    for in = 1:mesh.nn
        if mesh.bcn[in]==1
            x      = mesh.xn[in]
            y      = mesh.yn[in]
            vx, vy, p, Sxx, Syy, Sxy, sx, sy = EvalSolSevilla2018( x, y )
            Vx[in] = vx
            Vy[in] = vy
        end
    end
    for e = 1:mesh.nel
        x      = mesh.xc[e]
        y      = mesh.yc[e]
        vx, vy, p, Sxx, Syy, Sxy, sx, sy = EvalSolSevilla2018( x, y )
        Pa[e]   = p
        Sxxa[e] = Sxx   
        se[e,1] = sx
        se[e,2] = sy
    end
    for in = 1:mesh.nv
        if mesh.bcn[in]==1
            x      = mesh.xn[in]
            y      = mesh.yn[in]
            vx, vy, p, Sxx, Syy, Sxy, sx, sy = EvalSolSevilla2018( x, y )
            P[in] = p
        end
    end

    #-----------------------------------------------------------------#
    @time  K_all, Q_all, Mi_all, b_all = ElementAssemblyLoopFEM( se, mesh, ipx, ipw, N, dNdX )
    #-----------------------------------------------------------------#
    
    if USE_DIRECT
        @time M, b, K, Q, Qt, M0 = SparseAssembly( K_all, Q_all, Mi_all, b_all, mesh, Vx, Vy, P )
        @time DirectSolveFEM!( M, K, Q, Qt, M0, b, Vx, Vy, P, mesh, b )
    else
        nout    = 1000#1e1
        iterMax = 3e4
        ϵ_PT    = 1e-7
        # θ       = 0.11428 *1.7
        # Δτ      = 0.28 /1.2
        ΔVxΔτ = zeros(mesh.nn)
        ΔVyΔτ = zeros(mesh.nn)
        ΔVxΔτ0= zeros(mesh.nn)
        ΔVyΔτ0= zeros(mesh.nn)
        ΔPΔτ  = zeros(mesh.np)

        # PT loop
        local iter = 0
        success = 0
        @time while (iter<iterMax)
            iter  += 1
            ΔVxΔτ0 .= ΔVxΔτ 
            ΔVyΔτ0 .= ΔVyΔτ
            if USE_NODAL
                ResidualStokesNodalFEM!( ΔVxΔτ, ΔVyΔτ, ΔPΔτ, Vx, Vy, P, mesh, K_all, Q_all, b_all )
            else
                ResidualStokesElementalSerialFEM!( ΔVxΔτ, ΔVyΔτ, ΔPΔτ, Vx, Vy, P, mesh, K_all, Q_all, b_all )
            end
            ΔVxΔτ  .= (1.0 - θ).*ΔVxΔτ0 .+ ΔVxΔτ 
            ΔVyΔτ  .= (1.0 - θ).*ΔVyΔτ0 .+ ΔVyΔτ
            Vx    .+= ΔτV .* ΔVxΔτ
            Vy    .+= ΔτV .* ΔVyΔτ
            P     .+= ΔτP .* ΔPΔτ
            if iter % nout == 0 || iter==1
                errVx = norm(ΔVxΔτ)/sqrt(length(ΔVxΔτ))
                errVy = norm(ΔVyΔτ)/sqrt(length(ΔVyΔτ))
                errP  = norm(ΔPΔτ) /sqrt(length(ΔPΔτ))
                @printf("PT Iter. %05d:\n", iter)
                @printf("  ||Fx|| = %3.3e\n", errVx)
                @printf("  ||Fy|| = %3.3e\n", errVy)
                @printf("  ||Fp|| = %3.3e\n", errP )
                err = max(errVx, errVy, errP)
                if err < ϵ_PT
                    print("PT solve converged in ")
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
    # Compute strain rate and stress
    εxx = zeros(mesh.nel, nip)
    εyy = zeros(mesh.nel, nip)
    εxy = zeros(mesh.nel, nip)
    τxx = zeros(mesh.nel, nip)
    τyy = zeros(mesh.nel, nip)
    τxy = zeros(mesh.nel, nip)
    ComputeStressFEM!( εxx, εyy, εxy, τxx, τyy, τxy, Vx, Vy, mesh, ipx, ipw, N, dNdX ) 

    #-----------------------------------------------------------------#
    Vxe  = zeros(mesh.nel)
    Vye  = zeros(mesh.nel)
    Ve   = zeros(mesh.nel)
    Pe   = zeros(mesh.nel)
    Sxxe = zeros(mesh.nel)

    P .-= minimum(P) 
    for e=1:mesh.nel
        for i=1:mesh.nnel
            Vxe[e] += 1.0/mesh.nnel * Vx[mesh.e2n[e,i]]
            Vye[e] += 1.0/mesh.nnel * Vy[mesh.e2n[e,i]]
            Ve[e]  += 1.0/mesh.nnel * sqrt(Vx[mesh.e2n[e,i]]^2 + Vy[mesh.e2n[e,i]]^2)
        end
        for i=1:mesh.npel
            Pe[e] += 1.0/mesh.npel * P[mesh.e2p[e,i]]
        end
        for ip=1:nip
            Sxxe[e] += 1.0/nip * τxx[e,ip]
        end
    end
    Sxxe .-= Pe
    @printf("%2.2e %2.2e\n", minimum(Vx), maximum(Vx))
    @printf("%2.2e %2.2e\n", minimum(Vy), maximum(Vy))
    @printf("%2.2e %2.2e\n", minimum(P),  maximum(P) )

    #-----------------------------------------------------------------#
    if USE_MAKIE
        # PlotMakie(mesh, Ve, xmin, xmax, ymin, ymax; cmap=:jet1)
        PlotMakie(mesh, Sxxa, xmin, xmax, ymin, ymax; cmap=:jet1)
    else
        PlotPyPlot(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1 )
    end
end

# main(1, 7, 1, 6, 0.030598470000000003, 0.03666666667,  1.0) # nit = 4000
# main(2, 7, 1, 6, 0.030598470000000003/2, 0.03666666667,  1.0) # nit = 9000
main(1, 7, 3, 6, 0.030598470000000003, 0.03666666667,  1.0) # nit = 4000



