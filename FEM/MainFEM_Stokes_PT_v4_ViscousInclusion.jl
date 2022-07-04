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

function ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX ) # Adapted from MILAMIN_1.0.1
    ndof         = 2*mesh.nnel
    npel         = mesh.npel
    nip          = length(ipw)
    K_all        = zeros(mesh.nel, ndof, ndof) 
    Q_all        = zeros(mesh.nel, ndof, npel)
    Mi_all       = zeros(mesh.nel, npel, npel)
    b            = zeros(mesh.nel, ndof)
    K_ele        = zeros(ndof, ndof)
    Q_ele        = zeros(ndof, npel)
    M_ele        = zeros(npel,npel)
    M_inv        = zeros(npel,npel)
    b_ele        = zeros(ndof)
    B            = zeros(ndof,3)
    m            = [ 1.0; 1.0; 0.0]
    Dev          = [ 4/3 -2/3  0.0;
                    -2/3  4/3  0.0;
                     0.0  0.0  1.0]
    P  = ones(npel,npel)
    Pb = ones(npel)
    PF = 1e3*maximum(mesh.ke)

    # Element loop
    @inbounds for e = 1:mesh.nel
        nodes   = mesh.e2n[e,:]
        x       = [mesh.xn[nodes] mesh.yn[nodes]]  
        ke      = mesh.ke[e]
        J       = zeros(2,2)
        invJ    = zeros(2,2)
        K_ele  .= 0.0
        Q_ele  .= 0.0
        M_ele  .= 0.0
        b_ele  .= 0.0
        M_inv  .= 0.0
        if npel==3  P[2:3,:] .= (x[1:3,:])' end

        # Integration loop
        for ip=1:nip

            if npel==3 
                Ni       = N[ip,:,1]
                Pb[2:3] .= x'*Ni 
                Pi       = P\Pb
            end

            dNdXi     = dNdX[ip,:,:]
            J        .= x'*dNdXi
            detJ      = J[1,1]*J[2,2] - J[1,2]*J[2,1]
            invJ[1,1] = +J[2,2] / detJ
            invJ[1,2] = -J[1,2] / detJ
            invJ[2,1] = -J[2,1] / detJ
            invJ[2,2] = +J[1,1] / detJ
            dNdx      = dNdXi*invJ
            B[1:2:end,1] .= dNdx[:,1]
            B[2:2:end,2] .= dNdx[:,2]
            B[1:2:end,3] .= dNdx[:,2]
            B[2:2:end,3] .= dNdx[:,1]
            Bvol          = dNdx'
            K_ele .+= ipw[ip] .* detJ .* ke  .* (B*Dev*B')
            if npel==3 
                Q_ele   .-= ipw[ip] .* detJ .* (Bvol[:]*Pi') 
                M_ele   .+= ipw[ip] .* detJ .* Pi*Pi'
            elseif npel==1 
                Q_ele   .-= ipw[ip] .* detJ .* 1.0 .* (B*m*npel') 
            end
            # b_ele   .+= ipw[ip] .* detJ .* se[e] .* N[ip,:] 
        end
        if npel==3 M_inv .= inv(M_ele) end
        # if npel==3 K_ele .+=  PF.*(Q_ele*M_inv*Q_ele') end
        if npel==3 Mi_all[e,:,:] .= M_inv end
        K_all[e,:,:]  .= K_ele
        Q_all[e,:,:]  .= Q_ele
        # b[e,:]       .= b_ele
    end
    return K_all, Q_all, Mi_all, b
end 

#----------------------------------------------------------#

function SparseAssembly( K_all, Q_all, Mi_all, mesh, Vx, Vy )

    println("Assembly")
    ndof = mesh.nn*2
    npel = mesh.npel
    rhs  = zeros(ndof+mesh.nel*npel)
    I_K  = Int64[]
    J_K  = Int64[]
    V_K  = Float64[]
    I_Q  = Int64[]
    J_Q  = Int64[]
    V_Q  = Float64[]
    I_Qt = Int64[]
    J_Qt = Int64[]
    V_Qt = Float64[]

    # Assembly of global sparse matrix
    @inbounds for e=1:mesh.nel
        nodes   = mesh.e2n[e,:]
        nodesVx = mesh.e2n[e,:]
        nodesVy = nodesVx .+ mesh.nn 
        nodesP  = e
        jj = 1
        for j=1:mesh.nnel

            # Q: ∇ operator: BC for V equations
            if mesh.bcn[nodes[j]] != 1
                for i=1:npel
                    push!(I_Q, nodesVx[j]); push!(J_Q, nodesP+(i-1)*mesh.nel); push!(V_Q, Q_all[e,jj  ,i])
                    push!(I_Q, nodesVy[j]); push!(J_Q, nodesP+(i-1)*mesh.nel); push!(V_Q, Q_all[e,jj+1,i])
                end
            end

            # Qt: ∇⋅ operator: no BC for P equations
            for i=1:npel
                push!(J_Qt, nodesVx[j]); push!(I_Qt, nodesP+(i-1)*mesh.nel); push!(V_Qt, Q_all[e,jj  ,i])
                push!(J_Qt, nodesVy[j]); push!(I_Qt, nodesP+(i-1)*mesh.nel); push!(V_Qt, Q_all[e,jj+1,i])
            end

            if mesh.bcn[nodes[j]] != 1
                # If not Dirichlet, add connection
                ii = 1
                for i=1:mesh.nnel
                    if mesh.bcn[nodes[i]] != 1 # Important to keep matrix symmetric
                        push!(I_K, nodesVx[j]); push!(J_K, nodesVx[i]); push!(V_K, K_all[e,jj,  ii  ])
                        push!(I_K, nodesVx[j]); push!(J_K, nodesVy[i]); push!(V_K, K_all[e,jj,  ii+1])
                        push!(I_K, nodesVy[j]); push!(J_K, nodesVx[i]); push!(V_K, K_all[e,jj+1,ii  ])
                        push!(I_K, nodesVy[j]); push!(J_K, nodesVy[i]); push!(V_K, K_all[e,jj+1,ii+1])
                    else
                        rhs[nodesVx[j]] -= K_all[e,jj  ,ii  ]*Vx[nodes[i]]
                        rhs[nodesVx[j]] -= K_all[e,jj  ,ii+1]*Vy[nodes[i]]
                        rhs[nodesVy[j]] -= K_all[e,jj+1,ii  ]*Vx[nodes[i]]
                        rhs[nodesVy[j]] -= K_all[e,jj+1,ii+1]*Vy[nodes[i]]
                    end
                    ii+=2
                end
                # rhs[nodes[j]] += b[e,j]
            else
                # Deal with Dirichlet: set one on diagonal and value and RHS
                push!(I_K, nodesVx[j]); push!(J_K, nodesVx[j]); push!(V_K, 1.0)
                push!(I_K, nodesVy[j]); push!(J_K, nodesVy[j]); push!(V_K, 1.0)
                rhs[nodesVx[j]] += Vx[nodes[j]]
                rhs[nodesVy[j]] += Vy[nodes[j]]
            end
            jj+=2
        end 
    end
    K  = sparse(I_K,  J_K,  V_K, mesh.nn*2, mesh.nn*2)
    Q  = sparse(I_Q,  J_Q,  V_Q, ndof, mesh.nel*npel)
    Qt = sparse(I_Qt, J_Qt, V_Qt, mesh.nel*npel, ndof)
    M0 = sparse(Int64[], Int64[], Float64[], mesh.nel*npel, mesh.nel*npel)
    M  = [K Q; Qt M0]
    return M, rhs, K, Q, Qt, M0
end

#----------------------------------------------------------#

function DirectSolveFEM!( M, K, Q, Qt, M0, rhs, Vx, Vy, P, mesh, b)
    println("Direct solve")
    ndof       = mesh.nn*2
    npel       = mesh.npel
    sol        = zeros(ndof+mesh.nel*npel)
    M[end,:]  .= 0.0 # add one Dirichlet constraint on pressure
    M[end,end] = 1.0
    rhs[end]   = 0.0
    sol       .= M\rhs
    Vx        .= sol[1:length(Vx)] 
    Vy        .= sol[(length(Vx)+1):(length(Vx)+length(Vy))]
    println(size( P))
    println(( mesh.nel*npel))
    P         .= sol[(2*mesh.nn+1):end]
    return nothing
end

#----------------------------------------------------------#

function DirectSolveFEM_v0!( M, K, Q, Qt, M0, bu, bp, Vx, Vy, P, mesh, b)
    println("Direct solve")
    ndof       = mesh.nn*2
    sol        = zeros(ndof+mesh.np)
    rhs        = [bu; bp]
    M[end,:]  .= 0.0 # add one Dirichlet constraint on pressure
    M[end,end] = 1.0
    rhs[end]   = 0.0
    sol       .= M\rhs
    Vx        .= sol[1:length(Vx)] 
    Vy        .= sol[(length(Vx)+1):(length(Vx)+length(Vy))]
    P         .= sol[(ndof+1):end]
    return nothing
end

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
    solver     = 0 
    penalty    = 1e4
    tol        = 1e-9
    
    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)
  
    # Generate mesh
    mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, inclusion, R; nnel, npel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    println("Number of p nodes:  ", mesh.np)

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
        @time  K_all, Q_all, Mi_all, b = ElementAssemblyLoopFEM_v0( se, mesh, ipx, ipw, N, dNdX )
        @time M, bu, bp, K, Q, Qt, M0 = SparseAssembly_v0( K_all, Q_all, Mi_all, b,mesh, Vx, Vy, P )
        # @time DirectSolveFEM_v0!( M, K, Q, Qt, M0, bu, bp, Vx, Vy, P, mesh, b )
        
        # @time  K_all, Q_all, Mi_all, b = ElementAssemblyLoopFEM( se, mesh, ipw, N, dNdX )
        # @time M, rhs, K, Q, Qt, M0 = SparseAssembly( K_all, Q_all, Mi_all, mesh, Vx, Vy )
        @time DirectSolveFEM!( M, K, Q, Qt, M0, [bu;bp], Vx, Vy, P, mesh, b )
        # #-----------------------------------------------------------------## #-----------------------------------------------------------------#
        # @time Kuu, Kup, bu, bp = ElementAssemblyLoopFEM_v1( se, mesh, ipx, ipw, N, dNdX, Vx, Vy, P )
        # #-----------------------------------------------------------------#
        # @time StokesSolvers!(Vx, Vy, P, mesh, Kuu, Kup, bu, bp, Kuu, solver; penalty, tol)
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