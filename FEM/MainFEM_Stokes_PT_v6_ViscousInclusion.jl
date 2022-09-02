# This version uses functions from "FunctionsFEM_v2.jl"
# The FEM discretisation is precomputed, then reused (stiffness, residual evaluation, stress computation) 
# The code is written in defect correction form

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

include("FunctionsFEM_v2.jl")
include("FunctionsFEM.jl")
include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../SolversFCFV_Stokes.jl")
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

    V   = ( x=zeros(mesh.nn), y=zeros(mesh.nn) )       # Solution on nodes 
    P   = zeros(mesh.np)       # Solution on either elements or nodes 
    dV  = ( x=zeros(mesh.nn), y=zeros(mesh.nn) ) 
    dP  = zeros(mesh.np)       # Solution on either elements or nodes 
    se  = zeros(mesh.nel, 2)   # Source term on elements
    Pa  = zeros(mesh.nel)      # Solution on elements 
    ε   = ( xx=zeros(mesh.nel, nip), yy=zeros(mesh.nel, nip), xy=zeros(mesh.nel, nip) )
    τ   = ( xx=zeros(mesh.nel, nip), yy=zeros(mesh.nel, nip), xy=zeros(mesh.nel, nip) )
    ∇v  = zeros(mesh.nel, nip)
    ηip = zeros(mesh.nel, nip)
    for ip=1:nip
        ηip[:,ip] .= mesh.ke ./2
    end

    # Intial guess
    for in = 1:mesh.nn
        if mesh.bcn[in]==1
            x      = mesh.xn[in]
            y      = mesh.yn[in]
            vx, vy = EvalAnalDani( x, y, R, η[1], η[2] )
            V.x[in] = vx
            V.y[in] = vy
        end
    end
    for e = 1:mesh.nel
        x      = mesh.xc[e]
        y      = mesh.yc[e]
        vx, vy, p = EvalAnalDani( x, y, R, η[1], η[2] )
        Pa[e] = p
    end

    # Compute FEM discretisation
    dNdx, weight, sparsity, bc = Eval_dNdx_W_Sparsity( mesh, ipw, N, dNdX, V )
    @show keys(sparsity)
        
    #-----------------------------------------------------------------#
    ComputeStressFEM_v2!( ηip, ∇v, ε, τ, V, mesh, dNdx, weight ) 
    fu, fp = ResidualStokes_v2( bc, se, mesh, N, dNdx, weight,V, P, τ )

    #-----------------------------------------------------------------#
    @time Kuu, Kup, bu, bp = ElementAssemblyLoopFEM_v4( ηip, bc, sparsity, se, mesh, N, dNdx, weight, V, P, τ )

    #-----------------------------------------------------------------#
    @time StokesSolvers!(dV.x, dV.y, dP, mesh, Kuu, Kup, fu, fp, Kuu, solver; penalty, tol)
    V.x .+= dV.x
    V.y .+= dV.y
    P   .+= dP

    #-----------------------------------------------------------------#
    ComputeStressFEM_v2!( ηip, ∇v, ε, τ, V, mesh, dNdx, weight ) 
    fu, fp, nFx, nFy, nFp = ResidualStokes_v2( bc, se, mesh, N, dNdx, weight, V, P, τ )

    @printf("||Fx|| = %2.2e\n", nFx)
    @printf("||Fy|| = %2.2e\n", nFy)
    @printf("||Fp|| = %2.2e\n", nFp)

    #-----------------------------------------------------------------#
    Vxe  = zeros(mesh.nel)
    Vye  = zeros(mesh.nel)
    Ve   = zeros(mesh.nel)
    Pe   = zeros(mesh.nel)
    Sxxe = zeros(mesh.nel)

    for e=1:mesh.nel
        for i=1:mesh.nnel
            Vxe[e] += 1.0/mesh.nnel * V.x[mesh.e2n[e,i]]
            Vye[e] += 1.0/mesh.nnel * V.y[mesh.e2n[e,i]]
            Ve[e]  += 1.0/mesh.nnel * sqrt(V.x[mesh.e2n[e,i]]^2 + V.y[mesh.e2n[e,i]]^2)
        end
        for i=1:mesh.npel
            Pe[e]  += 1.0/mesh.npel * P[mesh.e2p[e,i]]
        end
        for ip=1:nip
            Sxxe[e] += 1.0/nip * τ.xx[e,ip]
        end
    end
    Sxxe .-= Pe
    @printf("min Vx %2.2e --- max. Vx %2.2e\n", minimum(V.x), maximum(V.x))
    @printf("min Vy %2.2e --- max. Vy %2.2e\n", minimum(V.y), maximum(V.y))
    @printf("min P  %2.2e --- min. P  %2.2e\n", minimum(P),  maximum(P) )
    @printf("min ∇v %2.2e --- min. ∇v %2.2e\n", minimum(∇v), maximum(∇v))

    #-----------------------------------------------------------------#
    PlotMakie(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1)
end

main(1, 7, 1, 6, 0.0382, 0.1833, 7.0) # nit = xxxxx