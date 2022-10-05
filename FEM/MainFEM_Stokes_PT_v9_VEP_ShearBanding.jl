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
using CellArrays, StaticArrays, Setfield

include("FunctionsFEM.jl")
include("FunctionsFEM_v2.jl")
include("FunctionsFEM_VEP_comp.jl")
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
    R          = 0.1
    inclusion  = 1
    εBG        = -1.0
    η0         = 1.0 
    G0         = 1.0 
    K0         = 2.0 
    ξ          = 10.0      # Maxwell relaxation time
    Δt         = η0/(G0*ξ + 1e-15)
    nt         = 1#16
    solver     = 1
    penalty    = 1e5
    tol        = 1e-10
    pl_params  = (C=1.6, cosϕ=1.0, sinϕ=sind(30), ηvp=1.2e-2, sinψ=0.0 )
    # pl_params  = (C=1.2, sinϕ=0*sind(30), ηvp=1.2e-2/2 )
    nitmax     = 20
    tol_abs    = 1e-10
    comp       = true
    
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

    mesh.ke[mesh.phase.==1] .= η0
    mesh.ke[mesh.phase.==2] .= η0

    V   = ( x=zeros(mesh.nn), y=zeros(mesh.nn) )       # Solution on nodes 
    P   = zeros(mesh.np)       # Solution on either elements or nodes 
    P0  = zeros(mesh.np)       # Solution on either elements or nodes 
    ΔP  = zeros(mesh.nel, nip) # Pressure corrections plasticity
    dV  = ( x=zeros(mesh.nn), y=zeros(mesh.nn) ) 
    dP  = zeros(mesh.np)       # Solution on either elements or nodes 
    se  = zeros(mesh.nel, 2)   # Source term on elements
    ε   = ( xx=zeros(mesh.nel, nip), yy=zeros(mesh.nel, nip), xy=zeros(mesh.nel, nip) )
    τ   = ( xx=zeros(mesh.nel, nip), yy=zeros(mesh.nel, nip), xy=zeros(mesh.nel, nip) )
    τ0  = ( xx=zeros(mesh.nel, nip), yy=zeros(mesh.nel, nip), xy=zeros(mesh.nel, nip) )
    ∇v  = zeros(mesh.nel, nip) 
    ηve = η0*ones(mesh.nel, nip)
    G   = G0*ones(mesh.nel, nip)
    K   = K0*ones(mesh.nel, nip)
    # Constitutive matrices
    celldims     = (4, 4)
    Cell         = SMatrix{celldims..., Float64, prod(celldims)}
    D_all        = CPUCellArray{Cell}(undef, mesh.nel, nip) 
    D_all.data  .= 0.0
    Dj_all       = CPUCellArray{Cell}(undef, mesh.nel, nip) 
    Dj_all.data .= 0.0

    # For postprocessing
    Fmom  = zeros(nitmax)
    Fcont = zeros(nitmax)
    τvec = zeros(nt)
    Vxe  = zeros(mesh.nel)
    Vye  = zeros(mesh.nel)
    Ve   = zeros(mesh.nel)
    Pe   = zeros(mesh.nel)
    τiie = zeros(mesh.nel)
    εiie = zeros(mesh.nel)
    
    # Intial guess
    for in = 1:mesh.nn
        x       = mesh.xn[in]
        y       = mesh.yn[in]
        V.x[in] = -x*εBG
        V.y[in] =  y*εBG
    end

    for ip=1:nip
        G[mesh.phase.==2, ip] .= G0/2
    end

    # Compute FEM discretisation
    dNdx, weight, sparsity, bc = Eval_dNdx_W_Sparsity( mesh, ipw, N, dNdX, V )
    
    #-----------------------------------------------------------------#
    for it=1:nt
        @printf("##### Time step %03d #####\n", it)

        Fmom  .= 0.0
        Fcont .= 0.0

        # Compute VE modulus
        for ip=1:nip
            ηve[:,ip] .= 1.0 ./( 1.0./(G[:,ip]*Δt) .+ 1.0./η0 ) 
        end

        # It would be nice to find a more elegant to unroll the tuple below
        τ0.xx .= τ.xx 
        τ0.yy .= τ.yy
        τ0.xy .= τ.xy
        P0    .= P

        k = 0
        for iter=1:nitmax

            k = iter
            @printf("##    Iteration %03d    ## (it=%03d)\n", iter, it)

            #-----------------------------------------------------------------#

            if comp==false
                ComputeStressFEM_v2!( D_all, pl_params, ηve, G, Δt, ∇v, ε, τ0, τ, V, P, mesh, dNdx, weight ) 
                fu, fp, nFx, nFy, nFp = ResidualStokes_v2( bc, se, mesh, N, dNdx, weight, V, P, τ )
            else
                # ComputeStressFEM_v2!( D_all, pl_params, ηve, G, Δt, ∇v, ε, τ0, τ, V, P, mesh, dNdx, weight ) 
                # Dj_all = D_all
                @time ComputeStressFEM_v3!( Dj_all, D_all, pl_params, ηve, G, K, Δt, ∇v, ε, τ0, τ, V, P0, P, ΔP, mesh, dNdx, weight ) 
                fu, fp, nFx, nFy, nFp = ResidualStokes_v3( bc, se, mesh, N, dNdx, weight, V, P0, P, ΔP, τ, K, Δt )
            end
            
            #-----------------------------------------------------------------#
            @printf("||Fx|| = %2.2e\n", nFx)
            @printf("||Fy|| = %2.2e\n", nFy)
            @printf("||Fp|| = %2.2e\n", nFp)
            Fmom[iter], Fcont[iter] = nFx, nFp
            if (nFx<tol_abs && nFy<tol_abs)# && nFp<tol_abs) 
                break
            end

            #-----------------------------------------------------------------#
            if comp==false
                @time Kuu, Kup, Kpu, Kpp, bu, bp = ElementAssemblyLoopFEM_v4( D_all, ηve, bc, sparsity, se, mesh, N, dNdx, weight, V, P, τ )
                @time StokesSolvers!(dV.x, dV.y, dP, mesh, Kuu, Kuu,  Kup, Kup,  Kpu, Kpp, fu, fp, Kuu, solver; penalty, tol)
            else
                @time Kuu, Kuuj, Kup, Kupj, Kpu, Kpp = ElementAssemblyLoopFEM_v5( Dj_all, D_all, ηve, K, Δt, bc, sparsity, se, mesh, N, dNdx, weight, V, P, τ )
                @time StokesSolvers!(dV.x, dV.y, dP, mesh, Kuuj, Kuu, Kupj, Kup, Kpu, Kpp, fu, fp, Kuu, solver; penalty, tol, comp)
            end

            V.x .+= dV.x
            V.y .+= dV.y
            P   .+= dP

        end
        # P .+= \Delta??
        
        p = Plots.plot( 1:k, log10.(Fmom[1:k]), title=it )
        p = Plots.plot!( 1:k, log10.(Fcont[1:k]), title=it )
        display(p)

        #-----------------------------------------------------------------#

        # For postprocessing
        τvec[it]     = abs(τ.xx[1,1])

        for e=1:mesh.nel
            Vxe[e]  = 0.0
            Vye[e]  = 0.0
            Ve[e]   = 0.0
            Pe[e]   = 0.0
            εiie[e] = 0.0
            τiie[e] = 0.0
            for i=1:mesh.nnel
                Vxe[e] += 1.0/mesh.nnel * V.x[mesh.e2n[e,i]]
                Vye[e] += 1.0/mesh.nnel * V.y[mesh.e2n[e,i]]
                Ve[e]  += 1.0/mesh.nnel * sqrt(V.x[mesh.e2n[e,i]]^2 + V.y[mesh.e2n[e,i]]^2)
            end
            for i=1:mesh.npel
                Pe[e] += 1.0/mesh.npel * P[mesh.e2p[e,i]]
            end
            for ip=1:nip
                εiie[e] += 1.0/nip * sqrt( 0.5*(ε.xx[e,ip]^2 + ε.yy[e,ip]^2) + ε.xy[e,ip]^2 )
                τiie[e] += 1.0/nip * sqrt( 0.5*(τ.xx[e,ip]^2 + τ.yy[e,ip]^2) + τ.xy[e,ip]^2 )
            end
        end
        @printf("min Vx  %2.2e --- max. Vx  %2.2e\n", minimum(V.x),  maximum(V.x))
        @printf("min Vy  %2.2e --- max. Vy  %2.2e\n", minimum(V.y),  maximum(V.y))
        @printf("min P   %2.2e --- min. P   %2.2e\n", minimum(P),    maximum(P) )
        @printf("min ∇v  %2.2e --- min. ∇v  %2.2e\n", minimum(∇v),   maximum(∇v))
        @printf("min τii %2.2e --- min. τii %2.2e\n", minimum(τiie), maximum(τiie) )
        @printf("min εii %2.2e --- min. εii %2.2e\n", minimum(εiie), maximum(εiie) )
        #-----------------------------------------------------------------#
        if inclusion==1 PlotMakie(mesh, εiie, xmin, xmax, ymin, ymax; cmap=:turbo) end
    end

    #-----------------------------------------------------------------#
    if inclusion==0 
        p = Plots.plot( 1:nt, τvec )
        display(p)
    end
end

main(2, 7, 1, 6, 0.0382, 0.1833, 7.0) # nit = xxxxx

#----------------------------------------------------------#