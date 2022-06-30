using Printf, DelimitedFiles, Plots

include("FunctionsFEM.jl")
include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("IntegrationPoints.jl")

#-----------------------------------------------------------------#

function main( n, nnel, npel, nip, θ, ΔτV, ΔτP )

    println("\n******** FEM STOKES ********")
    g = [0 -9.81]
    ρ = 917.0

    #-----------------------------------------------------------------#
    # Mesh generation from data file
    data = readdlm("./FEM/arolla51.txt", ' ')
    xp = zeros(2*size(data,1)) 
    yp = zeros(2*size(data,1))
    tp =  ones(Int64, 2*size(data,1))
    # Poulate arrays of boundary points with inpute data
    xp[1:size(data,1)]     .= data[:,1]
    xp[size(data,1)+1:end] .= reverse(data[:,1])
    yp[1:size(data,1)]     .= data[:,2]
    yp[size(data,1)+1:end] .= reverse(data[:,3])
    tp[1:size(data,1)]     .= 1
    tp[size(data,1)+2:end-1] .= 2
    # Add sliding section
    tp[xp.>2200 .&& xp.<2500] .= 2
    # Somehow need to subsample the data !!!
    xp = xp[2:2:end]
    yp = yp[2:2:end]
    tp = tp[2:2:end]
    # Check
    p = Plots.scatter( xp[tp.==1]./1e3, yp[tp.==1]./1e3, markershape =:cross, label="Base")
    p = Plots.scatter!( xp[tp.==2]./1e3, yp[tp.==2]./1e3, markershape =:cross, label="Surface")
    display(Plots.plot(p, ylims=(2,3.5), aspect_ratio=1.0, xlabel="x [km]", ylabel="y [km]"))
    # Generate
    model      = -1               
    area       = 200.0
    mesh = MakeTriangleMesh( 1, 1, minimum(xp), maximum(xp), minimum(yp), maximum(yp), 0.0, model, 0.0, 0, area; xp_in=xp, yp_in=yp, tp_in=tp, nnel=nnel, npel=npel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    println("Number of p nodes:  ", mesh.np)

    p = Plots.scatter!( mesh.xn[mesh.bcn.==1]./1e3, mesh.yn[mesh.bcn.==1]./1e3, markershape =:cross, label="Surface")
    p = Plots.scatter!( mesh.xn[mesh.bcn.==2]./1e3, mesh.yn[mesh.bcn.==2]./1e3, markershape =:cross, label="Surface")
    # Make surface points --> free surface
    mesh.bcn[mesh.bcn.==2] .= 0

    #-----------------------------------------------------------------#
    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)

    #-----------------------------------------------------------------#
    Vx  = zeros(mesh.nn)       # Solution on nodes 
    Vy  = zeros(mesh.nn)       # Solution on nodes 
    P   = zeros(mesh.np)       # Solution on either elements or nodes 
    se  = zeros(mesh.nel,2)    # Source on elements

    for e = 1:mesh.nel
        se[e,1] = ρ*g[1]
        se[e,2] = ρ*g[2]
    end

    #-----------------------------------------------------------------#
    @time  K_all, Q_all, Mi_all, b_all = ElementAssemblyLoopFEM( se, mesh, ipx, ipw, N, dNdX )

    #-----------------------------------------------------------------#
    @time M, b, K, Q, Qt, M0 = SparseAssembly( K_all, Q_all, Mi_all, b_all, mesh, Vx, Vy, P )
    @time DirectSolveFEM!( M, K, Q, Qt, M0, b, Vx, Vy, P, mesh, b )
    
    #-----------------------------------------------------------------#
    Vxe = zeros(mesh.nel)
    Vye = zeros(mesh.nel)
    Ve  = zeros(mesh.nel)
    Pe  = zeros(mesh.nel)
    for e=1:mesh.nel
        for i=1:mesh.nnel
            Vxe[e] += 1.0/mesh.nnel * Vx[mesh.e2n[e,i]]
            Vye[e] += 1.0/mesh.nnel * Vy[mesh.e2n[e,i]]
            Ve[e]  += 1.0/mesh.nnel * sqrt(Vx[mesh.e2n[e,i]]^2 + Vy[mesh.e2n[e,i]]^2)
        end
        for i=1:mesh.npel
            Pe[e] += 1.0/mesh.npel * P[mesh.e2p[e,i]]
        end
    end
    @printf("%2.2e %2.2e\n", minimum(Vx), maximum(Vx))
    @printf("%2.2e %2.2e\n", minimum(Vy), maximum(Vy))
    @printf("%2.2e %2.2e\n", minimum(P),  maximum(P) )

    #-----------------------------------------------------------------#
    PlotMakie(mesh, Pe, xmin, xmax, ymin, ymax; cmap=:turbo)

    #-----------------------------------------------------------------#
end

main( 1, 6, 1, 6, 0., 0., 0. )