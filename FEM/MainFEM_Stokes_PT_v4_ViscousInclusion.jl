const USE_GPU      = false  # Not supported yet 
const USE_DIRECT   = true   # Sparse matrix assembly + direct solver
const USE_NODAL    = false  # Nodal evaluation of residual
const USE_PARALLEL = false  # Parallel residual evaluation
const USE_MAKIE    = true   # Visualisation 
import Plots
import Statistics:mean

using Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid
using MAT

include("FunctionsFEM.jl")
include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../EvalAnalDani.jl")
include("IntegrationPoints.jl")

#----------------------------------------------------------#

function main( n, nnel, npel, nip, θ, ΔτV, ΔτP )

    println("\n******** FEM STOKES ********")
    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    nx, ny     = Int16(n*30), Int16(n*30)
    R          = 1.0
    inclusion  = 1
    εBG        = 1.0
    η          = [1.0 100.0] 
    solver     = -1 
    penalty    = 1e5
    tol        = 1e-9
    
    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)
  
    # Generate mesh
    mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R; nnel, npel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    println("Number of p nodes:  ", mesh.np)
    println("Min. bcn:  ", minimum(mesh.bcn))
    println("Max. bcn:  ", maximum(mesh.bcn))

    mesh.ke[mesh.phase.==1] .= η[1]
    mesh.ke[mesh.phase.==2] .= η[2]

    Vx = zeros(mesh.nn)       # Solution on nodes 
    Vy = zeros(mesh.nn)       # Solution on nodes 
    P  = zeros(mesh.np)       # Solution on either elements or nodes 
    se = zeros(mesh.nel, 2)   # Source term on elements
    Pa = zeros(mesh.nel)      # Solution on elements 

    # Intial guess
    for in = 1:mesh.nn
        if mesh.bcn[in]==1
            x      = mesh.xn[in]
            y      = mesh.yn[in]
            vx, vy = EvalAnalDani( x, y, R, η[1], η[2] )
            Vx[in] = vx
            Vy[in] = vy
        end
    end
    for e = 1:mesh.nel
        x      = mesh.xc[e]
        y      = mesh.yc[e]
        vx, vy, p = EvalAnalDani( x, y, R, η[1], η[2] )
        Pa[e] = p
    end
    
    # #-----------------------------------------------------------------#
    
    if USE_DIRECT
        #-----------------------------------------------------------------#
        @time K_all, Q_all, Mi_all, b, Kuu, Kup, bu, bp = ElementAssemblyLoopFEM_v1( se, mesh, ipx, ipw, N, dNdX, Vx, Vy, P )
        
        #-----------------------------------------------------------------#
        @time StokesSolvers!(Vx, Vy, P, mesh, Kuu, Kup, bu, bp, Kuu, solver; penalty, tol)
    else
        #-----------------------------------------------------------------#
         @time  K_all, Q_all, Mi_all, b = ElementAssemblyLoopFEM_v0( se, mesh, ipx, ipw, N, dNdX )
        nout    = 1000#1e1
        iterMax = 10e3#3e4
        ϵ_PT    = 5e-7
        ΔVxΔτ = zeros(mesh.nn)
        ΔVyΔτ = zeros(mesh.nn)
        ΔVxΔτ0= zeros(mesh.nn)
        ΔVyΔτ0= zeros(mesh.nn)
        ΔPΔτ  = zeros(mesh.np)

        #-----------------------------------------------#
        # Local Δτ for momentum equations (local Δτ for continuity does not seem to help)
        #-----------------------------------------------#
        ΔτVv  = zeros(mesh.nn)
        ΔτPv  = zeros(mesh.np)
        ηv    = zeros(mesh.nn)
        ηe    = zeros(mesh.nel)
        ηe   .= mesh.ke
        ΔτVv .= ΔτV
        ΔτPv .= ΔτP
        nludo = 1 # more than 1 does not seem to help
        itp   = 0 # only 0 and 3 seem to work

        for iludo=1:nludo
            # Compute nodal viscosities
            for i=1:mesh.nn
                n = 0
                η = 0.0
                for ii=1:length(mesh.n2e[i])
                    e       = mesh.n2e[i][ii]
                    n   += 1
                    if itp==3 η  = max(η, ηe[e]) end # Local maximum
                    if itp==0 η += ηe[e]         end # arithmetic mean
                    if itp==1 η += 1.0/ηe[e]     end # harmonic mean
                    if itp==2 η += log(ηe[e])    end # geometric mean
                end
                w = 1.0/n
                if itp==0 ηv[i] = w*η      end
                if itp==1 ηv[i] = w/η      end
                if itp==2 ηv[i] = exp(η)^w end
                if itp==3 ηv[i] = η        end
            end
            # Compute element viscosities
            for e=1:mesh.nel
                nodes = mesh.e2n[e,:]
                if itp==3 ηe[e] = max(ηv[nodes]...)  end
                if itp==0 ηe[e] = mean(ηv[nodes]) end
            end
        end
        ΔτVv ./= ηv
        #-----------------------------------------------#

        # PT loop
        local iter = 0
        success = 0
        @time while (iter<iterMax)
            iter  += 1
            ΔVxΔτ0 .= ΔVxΔτ 
            ΔVyΔτ0 .= ΔVyΔτ
            if USE_NODAL
                ResidualStokesNodalFEM!( ΔVxΔτ, ΔVyΔτ, ΔPΔτ, Vx, Vy, P, mesh, K_all, Q_all, b )
            else
                ResidualStokesElementalSerialFEM!( ΔVxΔτ, ΔVyΔτ, ΔPΔτ, Vx, Vy, P, mesh, K_all, Q_all, b )
            end
            ΔVxΔτ  .= (1.0 - θ).*ΔVxΔτ0 .+ ΔVxΔτ 
            ΔVyΔτ  .= (1.0 - θ).*ΔVyΔτ0 .+ ΔVyΔτ
            Vx    .+= ΔτVv .* ΔVxΔτ
            Vy    .+= ΔτVv .* ΔVyΔτ
            P     .+= ΔτPv .* ΔPΔτ
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
    ∇v  = zeros(mesh.nel, nip) 
    ComputeStressFEM!( ∇v, εxx, εyy, εxy, τxx, τyy, τxy, Vx, Vy, mesh, ipx, ipw, N, dNdX ) 

    #-----------------------------------------------------------------#
    Vxe  = zeros(mesh.nel)
    Vye  = zeros(mesh.nel)
    Ve   = zeros(mesh.nel)
    Pe   = zeros(mesh.nel)
    Sxxe = zeros(mesh.nel)

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
    @printf("min Vx %2.2e --- max. Vx %2.2e\n", minimum(Vx), maximum(Vx))
    @printf("min Vy %2.2e --- max. Vy %2.2e\n", minimum(Vy), maximum(Vy))
    @printf("min P  %2.2e --- min. P  %2.2e\n", minimum(P),  maximum(P) )
    @printf("min ∇v %2.2e --- min. ∇v %2.2e\n", minimum(∇v), maximum(∇v))

    #-----------------------------------------------------------------#
    if USE_MAKIE
        PlotMakie(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1)
    else
        PlotPyPlot(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1 )
    end
end

main(1, 7, 3, 6, 0.0382, 0.1833, 7.0) # nit = xxxxx