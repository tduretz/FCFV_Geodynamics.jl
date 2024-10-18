using Printf, DelimitedFiles, Plots, Interpolations, LinearAlgebra

include("FunctionsFEM.jl")
include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
include("../SolversFCFV_Stokes.jl")
include("IntegrationPoints.jl")
include("Helpers.jl")

#-----------------------------------------------------------------#
@views function read_data(dat_file::String; resol::Int=128, visu_chk::Bool=false)
    print("Reading the data... ")
    data    = readdlm(dat_file, Float64)
    data    = data[3:end-2,:]
    xv_d    = data[:,1]
    bed_d   = data[:,2]
    surf_d  = data[:,3]
    xv      = LinRange(xv_d[1], xv_d[end], resol)
    itp1    = interpolate((xv_d,), bed_d[:,1] , Gridded(Linear()))
    itp2    = interpolate((xv_d,), surf_d[:,1], Gridded(Linear()))
    bed     = itp1.(xv)[2:end-1] # DEBUG: crop first and last data points
    surf    = itp2.(xv)[2:end-1] # DEBUG: crop first and last data points
    xv      = xv[2:end-1]        # DEBUG: crop first and last data points
    dat_len = length(xv)
    @assert dat_len == size(bed)[1] == size(surf)[1]
    xp, yp  = zeros(2*dat_len), zeros(2*dat_len)
    tp      = ones(Int64,2*dat_len)
    # Populate arrays of boundary points with input data
    xp[1:dat_len]       .= xv
    xp[dat_len+1:end]   .= reverse(xv)
    yp[1:dat_len]       .= bed
    yp[dat_len+1:end]   .= reverse(surf)
    tp[1:dat_len]       .= 1
    tp[dat_len+2:end-1] .= 2
    # tp[xp.>2200 .&& xp.<2500] .= 2 # Add sliding section
    # Check
    if visu_chk
        p = Plots.scatter( xp[tp.==1]./1e3, yp[tp.==1]./1e3, markershape =:cross, label="Base")
        p = Plots.scatter!( xp[tp.==2]./1e3, yp[tp.==2]./1e3, markershape =:cross, label="Surface")
        display(Plots.plot(p, aspect_ratio=1.0, xlabel="x [km]", ylabel="y [km]"))
    end
    print("Interpolating original data (nx=$(size(bed_d)[1])) on nx=$(size(bed)[1]) grid... ")
    println("done.")
    return xp, yp, tp
end

#-----------------------------------------------------------------#

@views function main( n, nnel, npel, nip, θ, ΔτV, ΔτP )

    println("\n******** FEM STOKES ********")
    g      = [0.0 -1.0]
    ρ      = 1
    η      = 1.0
    solver = -1

    #-----------------------------------------------------------------#
    # Mesh generation from data file
    xp, yp, tp = read_data("./FEM/arolla51.txt"; resol=100, visu_chk=false)
    Lc     = maximum(xp) - minimum(xp)   
    xp, yp = xp./Lc, yp./Lc
    # Add sliding section
    tp[xp.>2200 .&& xp.<2500 .&& yp.<2700] .= 4
    # Check
    # p = Plots.scatter( xp[tp.==1]./1e3, yp[tp.==1]./1e3, markershape =:cross, label="Base")
    # p = Plots.scatter!( xp[tp.==2]./1e3, yp[tp.==2]./1e3, markershape =:cross, label="Surface")
    # p = Plots.scatter!( xp[tp.==4]./1e3, yp[tp.==4]./1e3, markershape =:cross, label="Free slip")
    # display(Plots.plot(p, ylims=(2,3.5), aspect_ratio=1.0, xlabel="x [km]", ylabel="y [km]"))

    # Generate
    model = -1               
    area  = 8/Lc^2
    # area  = 100/Lc^2
    mesh  = MakeTriangleMesh( 1, 1, minimum(xp), maximum(xp), minimum(yp), maximum(yp), 0.0, model, 0.0, 0, area; xp_in=xp, yp_in=yp, tp_in=tp, nnel=nnel, npel=npel ) 
    println("Number of elements: ", mesh.nel)
    println("Number of vertices: ", mesh.nn)
    println("Number of p nodes:  ", mesh.np)

    # Check
    # p = Plots.scatter( mesh.xn[mesh.bcn.==1]./1e3, mesh.yn[mesh.bcn.==1]./1e3, markershape =:cross, label="Base")
    # p = Plots.scatter!( mesh.xn[mesh.bcn.==2]./1e3, mesh.yn[mesh.bcn.==2]./1e3, markershape =:cross, label="Surface")
    # p = Plots.scatter!( mesh.xn[mesh.bcn.==4]./1e3, mesh.yn[mesh.bcn.==4]./1e3, markershape =:cross, label="Free slip")
    # # p = Plots.scatter!( mesh.xn./1e3, mesh.yn./1e3, markershape =:xcross, label="All")
    # display(Plots.plot(p, ylims=(2,3.5), aspect_ratio=1.0, xlabel="x [km]", ylabel="y [km]"))

    #-----------------------------------------------------------------#
    # Element data
    ipx, ipw  = IntegrationTriangle(nip)
    N, dNdX   = ShapeFunctions(ipx, nip, nnel)

    #-----------------------------------------------------------------#
    Vx  = zeros(mesh.nn)       # Solution on nodes 
    Vy  = zeros(mesh.nn)       # Solution on nodes 
    P   = zeros(mesh.np)       # Solution on either elements or nodes 
    se  = zeros(mesh.nel,2)    # Source on elements

    se[:,1]    .= ρ*g[1]
    se[:,2]    .= ρ*g[2]
    mesh.ke[:] .= η 
    
    #-----------------------------------------------------------------#
    @time Kuu, Kup, bu, bp = ElementAssemblyLoopFEM_v2( se, mesh, ipx, ipw, N, dNdX, Vx, Vy, P )
    
    #-----------------------------------------------------------------#
    coef  = 0 .*ones(mesh.nel*mesh.npel)
    Kpp   = spdiagm(coef)
    @time StokesSolvers!(Vx, Vy, P, mesh, Kuu, Kup, -Kup', Kpp, bu, bp, Kuu, solver; penalty=1e2)
    
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
    pminmax(Vx)
    pminmax(Vy)
    pminmax(P)

    #-----------------------------------------------------------------#
    @time PlotMakie(mesh, Pe, minimum(mesh.xn), maximum(mesh.xn), minimum(mesh.yn), maximum(mesh.yn); cmap=:turbo)

    #-----------------------------------------------------------------#
end

for i=1:1
   main( 1, 7, 1, 3, 0., 0., 0. )
end