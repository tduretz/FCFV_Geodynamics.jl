const USE_GPU      = false  # Not supported yet 
const USE_DIRECT   = true   # Sparse matrix assembly + direct solver
const USE_NODAL    = false  # Nodal evaluation of residual
const USE_PARALLEL = false  # Parallel residual evaluation
const USE_MAKIE    = true   # Visualisation 
import Plots
import Statistics:mean

using Printf, LinearAlgebra, SparseArrays, MAT, StaticArrays, Setfield
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid
using MAT

include("FunctionsFEM.jl")
include("FunctionsFEM_v2.jl")
include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../SolversFCFV_Stokes.jl")
include("IntegrationPoints.jl")

#----------------------------------------------------------#

function main( n, nnel, npel, nip, θ, ΔτV, ΔτP )

    println("\n******** FEM STOKES ********")
    # Create sides of mesh
    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
    nx, ny     = Int16(n*30), Int16(n*30)
    R          = 1.0
    inclusion  = 0
    εBG        = 1.0
    η0         = 1.0 
    G0         = 1.0 
    ξ          = 10.0      # Maxwell relaxation time
    Δt         =  η0/(G0*ξ + 1e-15)
    nt         = 20
    solver     = -1 
    penalty    = 1e5
    tol        = 1e-9
    pl_params  = (τ_y=1.6, sinϕ=sind(30), η_reg=1.2e-2 )
    nitmax     = 10
    tol_abs    = 1e-8
    
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

    mesh.phase = ones(Int, mesh.nel)
    mesh.ke[mesh.phase.==1] .= η0

    V   = ( x=zeros(mesh.nn), y=zeros(mesh.nn) )       # Solution on nodes 
    P   = zeros(mesh.np)       # Solution on either elements or nodes 
    dV  = ( x=zeros(mesh.nn), y=zeros(mesh.nn) ) 
    dP  = zeros(mesh.np)       # Solution on either elements or nodes 
    se  = zeros(mesh.nel, 2)   # Source term on elements
    ε   = ( xx=zeros(mesh.nel, nip), yy=zeros(mesh.nel, nip), xy=zeros(mesh.nel, nip) )
    τ   = ( xx=zeros(mesh.nel, nip), yy=zeros(mesh.nel, nip), xy=zeros(mesh.nel, nip) )
    τ0  = ( xx=zeros(mesh.nel, nip), yy=zeros(mesh.nel, nip), xy=zeros(mesh.nel, nip) )
    ∇v  = zeros(mesh.nel, nip) 
    ηv  = η0*ones(mesh.nel, nip)
    ηve = η0*ones(mesh.nel, nip)
    G   = G0*ones(mesh.nel, nip)
    
    # Intial guess
    for in = 1:mesh.nn
        if mesh.bcn[in]==1
            x      = mesh.xn[in]
            y      = mesh.yn[in]
            V.x[in] = -x*εBG
            V.y[in] =  y*εBG
        end
    end

    # Compute FEM discretisation
    dNdx, weight, sparsity, bc = Eval_dNdx_W_Sparsity( mesh, ipw, N, dNdX, V )
    @show keys(sparsity)

    # For postprocessing
    τvec     = zeros(nt)
    τvec_ana = zeros(nt)
    
    #-----------------------------------------------------------------#
    
    for it=1:nt
        @printf("##### Time step %03d #####\n", it)

        # Compue VE modulus
        mesh.ke .= 1.0 ./( 1.0/(G0*Δt) + 1.0/η0 ) 
        for ip=1:nip
            ηve[:,ip] .= mesh.ke
        end

        # It would be nice to find a more elegant to unroll the tuple below
        τ0.xx .= τ.xx 
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy

        for iter=1:nitmax
            @printf("##    Iteration %03d    ##\n", iter)

            #-----------------------------------------------------------------#
            ComputeStressFEM_v2!( pl_params, ηve, G, Δt, ∇v, ε, τ0, τ, V, P, mesh, dNdx, weight ) 
            fu, fp, nFx, nFy, nFp = ResidualStokes_v1( bc, sparsity, se, mesh, N, dNdx, weight, V, P, τ )
            
            @printf("||Fx|| = %2.2e\n", nFx)
            @printf("||Fy|| = %2.2e\n", nFy)
            @printf("||Fp|| = %2.2e\n", nFp)
            if (nFx<tol_abs && nFy<tol_abs && nFp<tol_abs) 
                @printf("Converged!\n")
                break
            end

            #-----------------------------------------------------------------#
            @time Kuu, Kup, bu, bp = ElementAssemblyLoopFEM_v4( bc, sparsity, se, mesh, N, dNdx, weight, V, P, τ )

            #-----------------------------------------------------------------#
            @time StokesSolvers!(dV.x, dV.y, dP, mesh, Kuu, Kup, fu, fp, Kuu, solver; penalty, tol)
            V.x .+= dV.x
            V.y .+= dV.y
            P   .+= dP

        end

        #-----------------------------------------------------------------#

        # For postprocessing
        τvec[it]     = abs(τ.xx[1,1])
        τvec_ana[it] = 2.0.*εBG.*η0.*(1.0.-exp.(.-(it*Δt).*G0./η0))
    end

    #-----------------------------------------------------------------#
    Vxe  = zeros(mesh.nel)
    Vye  = zeros(mesh.nel)
    Ve   = zeros(mesh.nel)
    Pe   = zeros(mesh.nel)
    τxxe = zeros(mesh.nel)

    for e=1:mesh.nel
        for i=1:mesh.nnel
            Vxe[e] += 1.0/mesh.nnel * V.x[mesh.e2n[e,i]]
            Vye[e] += 1.0/mesh.nnel * V.y[mesh.e2n[e,i]]
            Ve[e]  += 1.0/mesh.nnel * sqrt(V.x[mesh.e2n[e,i]]^2 + V.y[mesh.e2n[e,i]]^2)
        end
        for i=1:mesh.npel
            Pe[e] += 1.0/mesh.npel * P[mesh.e2p[e,i]]
        end
        for ip=1:nip
            τxxe[e] += 1.0/nip * sqrt( 0.5*(τ.xx[e,ip]^2 + τ.yy[e,ip]^2) + τ.xy[e,ip]^2 )
        end
    end
    @printf("min Vx  %2.2e --- max. Vx  %2.2e\n", minimum(V.x),  maximum(V.x))
    @printf("min Vy  %2.2e --- max. Vy  %2.2e\n", minimum(V.y),  maximum(V.y))
    @printf("min P   %2.2e --- min. P   %2.2e\n", minimum(P),   maximum(P) )
    @printf("min ∇v  %2.2e --- min. ∇v  %2.2e\n", minimum(∇v),  maximum(∇v))
    @printf("min τxx %2.2e --- min. τxx %2.2e\n", minimum(τ.xx), maximum(τ.xx) )

    #-----------------------------------------------------------------#

    p = Plots.plot( 1:nt, τvec )
    p = Plots.plot!( 1:nt, τvec_ana )
    display(p)
    # PlotMakie(mesh, τxxe, xmin, xmax, ymin, ymax; cmap=:turbo)

end

main(1, 7, 1, 6, 0.0382, 0.1833, 7.0) # nit = xxxxx