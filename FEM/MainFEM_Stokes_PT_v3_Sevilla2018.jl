const USE_GPU      = false  # Not supported yet 
const USE_DIRECT   = true   # Sparse matrix assembly + direct solver
const USE_NODAL    = false  # Nodal evaluation of residual
const USE_PARALLEL = false  # Parallel residual evaluation
const USE_MAKIE    = true   # Visualisation 
import Plots

using Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT#, StaticArrays
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid
import Statistics: mean
using MAT#, BenchmarkTools

include("FunctionsFEM.jl")
include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../SolversFCFV_Stokes.jl")
# include("../EvalAnalDani.jl")
include("IntegrationPoints.jl")

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
    nx, ny     = Int16(n*10), Int16(n*10)
    R          = 1.0
    inclusion  = 0
    εBG        = 1.0
    η          = [1.0 5.0] 
    BC         = [2; 1; 1; 1;] # S E N W --- 1: Dirichlet / 2: Neumann
    solver     = -1
    
    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)

    # Surface data
    ipx1D, ipw1D  = Integration1D(3)
    N1D, dNdX1D   = ShapeFunctions1D(ipx1D, 3, 3)
  
    # Generate mesh
    mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R, BC; nnel, npel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of nodes:    ", mesh.nn)
    println("Number of p nodes:  ", mesh.np)

    # flip
    # xc =  mesh.xn .- 0.5
    # xc .= -xc
    # mesh.xn .= xc .+ 0.5
    mesh.bcn[mesh.xn.≈0.0] .= 1
    mesh.bcn[mesh.xn.≈1.0] .= 1

    p = Plots.scatter( mesh.xn[mesh.bcn.==1], mesh.yn[mesh.bcn.==1], markershape =:cross, label="Base")
    p = Plots.scatter!( mesh.xn[mesh.bcn.==2], mesh.yn[mesh.bcn.==2], markershape =:cross, label="Surface")
    display(Plots.plot(p, aspect_ratio=1.0, xlabel="x [km]", ylabel="y [km]"))

    Vx   = zeros(mesh.nn)       # Solution on nodes 
    Vy   = zeros(mesh.nn)       # Solution on nodes 
    σxx  = zeros(mesh.nn)       # Solution on nodes 
    σyy  = zeros(mesh.nn)       # Solution on nodes 
    σxy  = zeros(mesh.nn)       # Solution on nodes 
    P    = zeros(mesh.np)       # Solution on either elements or nodes 
    se   = zeros(mesh.nel, 2)   # Source on elements 
    Pa   = zeros(mesh.nel)      # Solution on elements 
    Sxxa = zeros(mesh.nel)      # Solution on elements 

    # Intial guess
    for in = 1:mesh.nn
        x      = mesh.xn[in]
        y      = mesh.yn[in]
        vx, vy, p, Sxx, Syy, Sxy, sx, sy = EvalSolSevilla2018( x, y )
        if mesh.bcn[in]==1
            Vx[in] = vx
            Vy[in] = vy
        end
        if mesh.bcn[in]==2 # Store nodal boundary stress
            σxx[in] = Sxx
            σyy[in] = Syy
            σxy[in] = Sxy
        end
    end
    for e = 1:mesh.nel
        x       = mesh.xc[e]
        y       = mesh.yc[e]
        vx, vy, p, Sxx, Syy, Sxy, sx, sy = EvalSolSevilla2018( x, y )
        Pa[e]   = p
        Sxxa[e] = Sxx   
        se[e,1] = sx
        se[e,2] = sy
    end
  
    #-----------------------------------------------------------------#
    @time Kuu, Kup, bu, bp = ElementAssemblyLoopFEM_v2( se, mesh, ipx, ipw, N, dNdX, Vx, Vy, P )
    

    #----------------------------- SURFACE TRACTIONS -----------------------------#
    nnfa  = 3                  # number of nodes per face
    nipfa = 3                  # number of integration point per face
    nA    = [ 3 1 2 ]          # left node
    nB    = [ 4 5 6 ]          # mid-face node equivalent to mesh.e2f[e,f]
    nC    = [ 2 3 1 ]          # right node
    f2n   = zeros(Int64,3)     # face nodes to element nodes numbering
    nodes = zeros(Int64,3)     # global node numbering
    bct   = zeros(Int64,3)     # BC type (2 is Neumann)
    x     = zeros(nnfa,2)      # global coordinate of face nodes
    Ni    = zeros(nnfa)        # shape function for integration point
    dNdXi = ones(nnfa,2)       # shape function derivatives for integration point
    σ     = zeros(nnfa,3)      # Total stress tensor components for each node on the face
    T     = zeros(nnfa,2)      # Traction vector for each node on the face
    J     = zeros(2,2)         # Jacobian
    Fex   = zeros(mesh.nnel)   # Neumann force vector for the current element 
    Fey   = zeros(mesh.nnel)   # Neumann force vector for the current element 
    Fs    = zeros(2)           # Surface traction 
    Ff    = zeros(2*nnfa)      # Contribution of surface tratction to surface nodes
    # Loop over element (ideally one could loop only through elements that have Neumann nodes
    for e=1:1#mesh.nel  
        # Loop over faces of each element
        for f=1:mesh.nf_el 
            # Face normal
            nx         = mesh.n_x[e,f]
            ny         = mesh.n_y[e,f]
            # Node indices along the face (3 nodes since we only use quadratic elements)
            n1, n2, n3 = mesh.e2n[e,nA[f]], mesh.e2n[e,nB[f]],  mesh.e2n[e,nC[f]]  # mesh.e2f[e,f]
            nodes     .= [ n1,    n2,    n3    ]
            f2n       .= [ nA[f], nB[f], nC[f] ]
            bct       .= mesh.bcn[nodes]
            # Node coordinates
            x[:,1]    .= mesh.xn[nodes]      
            x[:,2]    .= mesh.yn[nodes]  
            # Nodal stresses
            σ[:,1]    .= σxx[nodes]
            σ[:,2]    .= σyy[nodes]
            σ[:,3]    .= σxy[nodes]
            # Nodal tractions
            T[:,1]    .= [σ[1,1]*nx+σ[1,3]*ny,  σ[2,1]*nx+σ[2,3]*ny,  σ[3,1]*nx+σ[3,3]*ny] .* (bct.==2) # only activate if BC is Neumann!!!
            T[:,2]    .= [σ[1,2]*ny+σ[1,3]*nx,  σ[2,2]*ny+σ[2,3]*nx,  σ[3,2]*ny+σ[3,3]*nx] .* (bct.==2)
            detJ  = abs((mesh.xn[n3]-mesh.xn[n1]) + (mesh.yn[n3]-mesh.yn[n1]))/2
            # Loop over integration loop
            for ip=1:nipfa 
                Ni         .= N1D[ip,:,:]
                dNdXi[:,1] .= dNdX1D[ip,:,:] # the second column is set to 1.0 by default dNdy = 1.0 to avoid detJ=0
                mul!(J, x', dNdXi)
                # detJ        = J[1,1]*J[2,2] - J[1,2]*J[2,1]
                # println(detJ, ' ', detJ1)
                # Surface tractions 
                Fs[1] = Ni'*T[:,1]  # use shape functions to evaluate traction at integration point
                Fs[2] = Ni'*T[:,2]
                # Integrate surface traction contributions
                Ff[1:nnfa]     .+= ipw1D[ip] .* Ni*Fs[1]*detJ
                Ff[nnfa+1:end] .+= ipw1D[ip] .* Ni*Fs[2]*detJ   
            end
            # Add contributions to element vector (sum?)
            Fex[f2n] .+= Ff[1:nnfa]
            Fey[f2n] .+= Ff[nnfa+1]
        end
        # Add contributions of element vector to global vector 
        e2n                = mesh.e2n[e,:]
        bcn                = mesh.bcn[e2n]
        if any(i->i==2, bcn)
            display(Fex)
            display(Fey)
        end
        bu[e2n]          .+= Fex
        bu[e2n.+mesh.nn] .+= Fey
    end
    #----------------------------- SURFACE TRACTIONS -----------------------------#

    #-----------------------------------------------------------------#

    if USE_DIRECT
        #-----------------------------------------------------------------#
        println("v2")
        @time Kuu, Kup, bu, bp = ElementAssemblyLoopFEM_v2( se, mesh, ipx, ipw, N, dNdX, Vx, Vy, P )

        #-----------------------------------------------------------------#
        @time StokesSolvers!(Vx, Vy, P, mesh, Kuu, Kup, bu, bp, Kuu, solver)
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

for i=1:1
    @time main(1, 7, 3, 6, 0.030598470000000003, 0.03666666667,  1.0) # nit = 4000
end