import UnicodePlots, Plots
using  Revise, Printf, LoopVectorization, LinearAlgebra, SparseArrays, MAT, Base.Threads

include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV_Stokes.jl")
include("SolversFCFV_Stokes.jl")
include("EvalAnalDani_v2.jl")
include("DiscretisationFCFV_Stokes_o2.jl")
BLAS.set_num_threads(4)
# include("MarkerRoutines.jl") 

#--------------------------------------------------------------------#

mutable struct Markers
    x         ::  Array{Float64,1}
    y         ::  Array{Float64,1}
    phase     ::  Array{Float64,1}
    cellx     ::  Array{Int64,1}#Vector{CartesianIndex{2}}
    celly     ::  Array{Int64,1}
    nmark     ::  Int64
    nmark_max ::  Int64
end

@views function LocateMarkers(p,dx,dy,xc,yc,xmin,xmax,ymin,ymax)
    # Find marker cell indices
    @threads for k=1:p.nmark
        if (p.x[k]<xmin || p.x[k]>xmax || p.y[k]<ymin || p.y[k]>ymax) 
            p.phase[k] = -1
        end
        if p.phase[k]>=0
            dstx         = p.x[k] - xc[1]
            i            = ceil(Int, dstx / dx + 0.5)
            dsty         = p.y[k] - yc[1]
            j            = ceil(Int, dsty / dy + 0.5)
            p.cellx[k]   = i
            p.celly[k]   = j
        end
    end
end

@views function Markers2Cells2(p,phase,xc,yc,dx,dy,ncx,ncy,prop,avg)
    weight      = zeros(Float64, (ncx, ncy))
    phase_th    = [similar(phase) for _ = 1:nthreads()] # per thread
    weight_th   = [similar(weight) for _ = 1:nthreads()] # per thread
    @threads for tid=1:nthreads()
        fill!(phase_th[tid] , 0)
        fill!(weight_th[tid], 0)
    end
    chunks = Iterators.partition(1:p.nmark, p.nmark ÷ nthreads())
    @sync for chunk in chunks
        Threads.@spawn begin
            tid = threadid()
            # fill!(phase_th[tid], 0)  # DON'T
            # fill!(weight_th[tid], 0)
            for k in chunk
                if p.phase[k]>=0
                # Get the indices:
                i = p.cellx[k]
                j = p.celly[k]
                # Relative distances
                dxm = 2.0 * abs(xc[i] - p.x[k])
                dym = 2.0 * abs(yc[j] - p.y[k])
                # Increment cell counts
                area = (1.0 - dxm / dx) * (1.0 - dym / dy)
                val  =  prop[Int64(p.phase[k])]
                if avg==0 phase_th[tid][i,  j] += val       * area end
                if avg==1 phase_th[tid][i,  j] += (1.0/val) * area end
                if avg==2 phase_th[tid][i,  j] += log(val) * area end
                weight_th[tid][i, j] += area
                end
            end
        end
    end
    phase  .= reduce(+, phase_th)
    weight .= reduce(+, weight_th)
    phase ./= weight
    if avg==1
        phase .= 1.0 ./ phase
    end
    if avg==2
        phase .= exp.(phase)
    end
    return
end

function SetMarkers!( p, R )
@tturbo for k=1:p.nmark
    in         = (p.x[k]^2 + p.y[k]^2) < R^2 
    p.phase[k] = (in==0)* 1.0 + (in==1)*2.0
end
end

#--------------------------------------------------------------------#

function SetUpProblem!(mesh, P, Vx, Vy, Sxx, Syy, Sxy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, sx, sy, rc, eta, gbar)
    # Evaluate T analytic on cell faces
    etam, etac = eta[1], eta[2]
    gr  = 0;                        # Simple shear: gr=1, er=0
    er  = -1;                       # Strain rate
    @tturbo for ifac=1:mesh.nf
        x, y                           = mesh.xf[ifac], mesh.yf[ifac]
        in                             = sqrt(x^2.0 + y^2.0)<=rc
        pa                             = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
        vxa, vya                       = SolutionFields_v( etam, etac, rc, er, gr, x, y, in)
        VxDir[ifac] = vxa
        VyDir[ifac] = vya
        # Pseudo-tractions
        dvxdxa, dvxdya, dvydxa, dvydya = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
        SxxNeu[ifac] = - pa +  etam*dvxdxa 
        SyyNeu[ifac] = - pa +  etam*dvydya
        SxyNeu[ifac] =         etam*dvxdya
        SyxNeu[ifac] =         etam*dvydxa
    end
    # Evaluate T analytic on barycentres
    @tturbo for iel=1:mesh.nel
        x, y                           = mesh.xc[iel], mesh.yc[iel]
        in                             = sqrt(x^2.0 + y^2.0)<=rc
        pa                             = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
        vxa, vya                       = SolutionFields_v( etam, etac, rc, er, gr, x, y, in)
        dvxdxa, dvxdya, dvydxa, dvydya = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
        P[iel]       = pa
        Vx[iel]      = vxa
        Vy[iel]      = vya
        etaa         = (in==0) * etam + (in==1) * etac
        Sxx[iel]     = -pa + 2.0*etaa*dvxdxa
        Syy[iel]     = -pa + 2.0*etaa*dvydya
        Sxy[iel]     = etaa*(dvxdya+dvydxa)
        sx[iel]      = 0.0
        sy[iel]      = 0.0
        out          = mesh.phase[iel] == 1.0
        mesh.ke[iel] = (out==1) * 1.0*etam + (out!=1) * 1.0*etac
    end

    # Compute jump condition
    @tturbo for iel=1:mesh.nel
        x, y                           = mesh.xc[iel], mesh.yc[iel] 
        for ifac=1:mesh.nf_el
            # Face
            nodei  = mesh.e2f[iel,ifac]
            nodei  = mesh.e2f[iel,ifac]
            ni_x   = mesh.n_x[iel,ifac]
            ni_y   = mesh.n_y[iel,ifac]
            phase1 = Int64(mesh.phase[iel])
            in     = (phase1==1) * 0 + (phase1==2) * 1 
            pre1                           = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
            dVxdx1, dVxdy1, dVydx1, dVydy1 = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
            eta_face1 = (in==0) * etam + (in==1) * etac
            tL_x  = ni_x*(-pre1 + eta_face1*dVxdx1) + ni_y*eta_face1*dVxdy1
            tL_y  = ni_y*(-pre1 + eta_face1*dVydy1) + ni_x*eta_face1*dVydx1
            # From element 2
            ineigh = (mesh.e2e[iel,ifac]>0) * mesh.e2e[iel,ifac] + (mesh.e2e[iel,ifac]<1) * iel
            phase2 = Int64(mesh.phase[ineigh])
            in     = (phase2==1) * 0 + (phase2==2) * 1 
            pre2                           = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
            dVxdx2, dVxdy2, dVydx2, dVydy2 = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
            eta_face2 = (in==0) * etam + (in==1) * etac
            tR_x = ni_x*(-pre2 + eta_face2*dVxdx2) + ni_y*eta_face2*dVxdy2
            tR_y = ni_y*(-pre2 + eta_face2*dVydy2) + ni_x*eta_face2*dVydx2
            gbar[iel,ifac,1] = 0.5 * (tL_x - tR_x)
            gbar[iel,ifac,2] = 0.5 * (tL_y - tR_y)
        end          
    end
    return
end

#--------------------------------------------------------------------#

function ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe, rc, eta )
    etam, etac = eta[1], eta[2]
    gr  = 0;                        # Simple shear: gr=1, er=0
    er  = -1;                       # Strain rate
    eVx  = zeros(mesh.nel)
    eVy  = zeros(mesh.nel)
    eTxx = zeros(mesh.nel)
    eTyy = zeros(mesh.nel)
    eTxy = zeros(mesh.nel)
    eP   = zeros(mesh.nel)
    Vxa  = zeros(mesh.nel)
    Vya  = zeros(mesh.nel)
    Txxa = zeros(mesh.nel)
    Tyya = zeros(mesh.nel)
    Txya = zeros(mesh.nel)
    Pa   = zeros(mesh.nel)
    Tiia = zeros(mesh.nel)
    Va   = zeros(mesh.nel)
    eV   = zeros(mesh.nel)
    eTii = zeros(mesh.nel)
    @tturbo for iel=1:mesh.nel
        x, y                           = mesh.xc[iel], mesh.yc[iel] 
        in                             = sqrt(x^2.0 + y^2.0)<=rc
        pa                             = SolutionFields_p( etam, etac, rc, er, gr, x, y, in)
        vxa, vya                       = SolutionFields_v( etam, etac, rc, er, gr, x, y, in)
        dvxdxa, dvxdya, dvydxa, dvydya = SolutionFields_dv(etam, etac, rc, er, gr, x, y, in)
        Pa[iel], Vxa[iel], Vya[iel]    = pa, vxa, vya
        Txxa[iel] = 2.0*mesh.ke[iel]*dvxdxa
        Tyya[iel] = 2.0*mesh.ke[iel]*dvydya 
        Txya[iel] = 1.0*mesh.ke[iel]*(dvxdya + dvydxa) 
        eVx[iel]  = Vxe[iel]  - Vxa[iel]
        eVy[iel]  = Vye[iel]  - Vya[iel]
        eTxx[iel] = Txxe[iel] - Txxa[iel]
        eTyy[iel] = Tyye[iel] - Tyya[iel]
        eTxy[iel] = Txye[iel] - Txya[iel]
        eP[iel]   = Pe[iel]   - Pa[iel]
        Va[iel]   = sqrt(Vxa[iel]^2 + Vya[iel]^2) 
        Ve        = sqrt(Vxe[iel]^2 + Vye[iel]^2)
        Tiia[iel] = sqrt(0.5*(Txxa[iel]^2 + Tyya[iel]^2) + Txya[iel]^2)
        Tiie      = sqrt(0.5*(Txxe[iel]^2 + Tyye[iel]^2) + Txye[iel]^2)
        eV[iel]   = Ve   - Va[iel]
        eTii[iel] = Tiie - Tiia[iel] 
    end
    errVx  = norm(eVx) /norm(Vxa)
    errVy  = norm(eVy) /norm(Vya)
    errTxx = norm(eTxx)/norm(Txxa)
    errTyy = norm(eTyy)/norm(Tyya)
    errTxy = norm(eTxy)/norm(Txya)
    errP   = norm(eP)  /norm(Pa)
    errV   = norm(eV)  /norm(Va)
    errTii = norm(eTii)/norm(Tiia)
    return errVx, errVy, errTxx, errTyy, errTxy, errP, errV, errTii, Txxa, Tyya, Txya
end

#--------------------------------------------------------------------#
    
function StabParam(τr, dA, Vol, mesh_type, coeff)
    # if mesh_type=="Quadrangles";        taui = coeff*τ  end
    if mesh_type=="Quadrangles";        τi = coeff*τr/dA  end
    if mesh_type=="UnstructTriangles";  τi = coeff*τr*dA  end
    return τi
end

#--------------------------------------------------------------------#

@views function main(n, mesh_type, order, new)

    println("\n******** FCFV STOKES ********")

    # Create sides of mesh
    xmin, xmax = -3.0, 3.0
    ymin, ymax = -3.0, 3.0
    # n          = 1
    nx, ny     = n, n
    solver     = 0
    R          = 1.0
    inclusion  = 1
    eta        = [1.0 100.0]
    # mesh_type  = "Quadrangles"
    # mesh_type  = "UnstructTriangles"
    BC         = [2; 1; 1; 1] # S E N W --- 1: Dirichlet / 2: Neumann
    o2         = order-1
    nmx        = 4            # marker per cell in x
    nmy        = 4            # marker per cell in y
    # Generate mesh
    if mesh_type=="Quadrangles" 
        # tau  = 1.0
        # if o2==0 tau  = 5e0 end
        # if o2==1 tau  = 1e1 end
        if o2==0 τr  = 6e-3 end
        if o2==1 τr  = 6e-3 end    
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R, BC )
    elseif mesh_type=="UnstructTriangles"  
        if o2==0 τr  = 1.0 end
        if o2==1 τr  = 1e4 end  
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, τr, inclusion, R, BC )
        # area = 1.0 # area factor: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        # ninc = 29  # number of points that mesh the inclusion: SETTING REPRODUCE THE RESULTS OF MATLAB CODE USING TRIANGLE
        # mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC, area, ninc ) 
    end
    println("Number of elements: ", mesh.nel, " --- Number of dofs: ", 2*mesh.nf+mesh.nel, " --- τr =  ", τr)

    # Initialise markers
    ncx, ncy  = nx, ny
    nmark0    = ncx*ncy*nmx*nmy; # total initial number of marker in grid
    dx,  dy   = (xmax-xmin)/ncx, (ymax-ymin)/ncy
    dxm, dym  = dx/nmx, dy/nmy 
    xm1d      =  LinRange(xmin+dxm/2, xmax-dxm/2, ncx*nmx)
    ym1d      =  LinRange(ymin+dym/2, ymax-dym/2, ncy*nmy)
    (xmi,ymi) = ([x for x=xm1d,y=ym1d], [y for x=xm1d,y=ym1d])
    xc        =  LinRange(xmin+dx/2, xmax-dx/2, ncx)
    yc        =  LinRange(ymin+dy/2, ymax-dy/2, ncy)
    # Over allocate markers
    nmark_max = 1*nmark0;
    phm    = zeros(Float64, nmark_max)
    xm     = zeros(Float64, nmark_max) 
    ym     = zeros(Float64, nmark_max)
    cellxm = zeros(Int64,   nmark_max)
    cellym = zeros(Int64,   nmark_max)
    xm[1:nmark0]     = vec(xmi)
    ym[1:nmark0]     = vec(ymi)
    phm[1:nmark0]    = zeros(Float64, size(xmi))
    cellxm[1:nmark0] = zeros(Int64,   size(xmi)) #zeros(CartesianIndex{2}, size(xm))
    cellym[1:nmark0] = zeros(Int64,   size(xmi))
    p      = Markers( xm, ym, phm, cellxm, cellym, nmark0, nmark_max )

    # Define phase
    SetMarkers!( p, R )

    # Update cell info on markers
    LocateMarkers(p,dx,dy,xc,yc,xmin,xmax,ymin,ymax)

    # Source term and BCs etc...
    Pa     = zeros(mesh.nel)
    Vxa    = zeros(mesh.nel)
    Vya    = zeros(mesh.nel)
    Sxxa   = zeros(mesh.nel)
    Syya   = zeros(mesh.nel)
    Sxya   = zeros(mesh.nel)
    sex    = zeros(mesh.nel)
    sey    = zeros(mesh.nel)
    VxDir  = zeros(mesh.nf)
    VyDir  = zeros(mesh.nf)
    SxxNeu = zeros(mesh.nf)
    SyyNeu = zeros(mesh.nf)
    SxyNeu = zeros(mesh.nf)
    SyxNeu = zeros(mesh.nf)
    gbar   = zeros(mesh.nel,mesh.nf_el,2)
    println("Model configuration:")
    @time SetUpProblem!(mesh, Pa, Vxa, Vya, Sxxa, Syya, Sxya, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, sex, sey, R, eta, gbar)


    k2d = ones(ncx,ncy)
    @time Markers2Cells2(p,k2d,xc,yc,dx,dy,ncx,ncy,eta,1)
    if mesh_type=="Quadrangles" 
         mesh.ke .= k2d[:]
    end

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    @time ae, be, be_o2, ze, pe, mei, pe, rjx, rjy = ComputeFCFV_o2(mesh, sex, sey, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, o2)

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    @time Kuu, Muu, Kup, fu, fp, tsparse = ElementAssemblyLoop_o2(mesh, ae, be, be_o2, ze, mei, pe, rjx, rjy, VxDir, VyDir, SxxNeu, SyyNeu, SxyNeu, SyxNeu, gbar, o2, new)

    # Solve for hybrid variable
    println("Linear solve:")
    @time Vxh, Vyh, Pe = StokesSolvers(mesh, Kuu, Kup, fu, fp, Muu, solver)

    # # Reconstruct element values
    println("Compute element values:")
    # @time Vxe, Vye, Txxe, Tyye, Txye = ComputeElementValues(mesh, Vxh, Vyh, Pe, ae, be, ze, VxDir, VyDir)
    @time Vxe, Vye, Txxe, Tyye, Txye = ComputeElementValues_o2(mesh, Vxh, Vyh, Pe, ae, be, be_o2, ze, rjx, rjy, mei, VxDir, VyDir, o2)

    # # Compute discretisation errors
    err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii, Txxa, Tyya, Txya = ComputeError( mesh, Vxe, Vye, Txxe, Tyye, Txye, Pe, R, eta )
    @printf("Error in Vx : %2.2e\n", err_Vx )
    @printf("Error in Vy : %2.2e\n", err_Vy )
    @printf("Error in Txx: %2.2e\n", err_Txx)
    @printf("Error in Tyy: %2.2e\n", err_Tyy)
    @printf("Error in Txy: %2.2e\n", err_Txy)
    @printf("Error in P  : %2.2e\n", err_P  )

    Perr = abs.(Pa.-Pe) 
    Verr = sqrt.( (Vxe.-Vxa).^2 .+ (Vye.-Vya).^2 ) 
    println("L_inf P error: ", maximum(Perr), " --- L_inf V error: ", maximum(Verr))

    # Visualise
    # println("Visualisation:")
    # PlotMakie(mesh, v, xmin, xmax, ymin, ymax; cmap = :viridis, min_v = minimum(v), max_v = maximum(v))
    # @time PlotMakie( mesh, Vxe, xmin, xmax, ymin, ymax, :jet1, minimum(Vxe), maximum(Vxe) )
    # @time PlotMakie( mesh, Verr, xmin, xmax, ymin, ymax, :jet1, minimum(Verr), maximum(Verr) )
    @time PlotMakie( mesh, Pe, xmin, xmax, ymin, ymax; cmap=:jet1, min_v=minimum(Pa), max_v=maximum(Pa) )
    # @time PlotMakie( mesh, Perr, xmin, xmax, ymin, ymax, :jet1, minimum(Perr), maximum(Perr) )
    # @time PlotMakie( mesh, Txxe, xmin, xmax, ymin, ymax, :jet1, -6.0, 2.0 )
    # @time PlotMakie( mesh, (mesh.ke), xmin, xmax, ymin, ymax, :jet1 )
    # @time PlotMakie( mesh, mesh.phase, xmin, xmax, ymin, ymax, :jet1)

    ndof = 2*mesh.nf+mesh.nel
    return ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii#, maximum(Perr), maximum(Verr)
end






#################### ORDER 1
new   = 1
order = 1 
N             = 30 .* [1; 2; 3; 4]#;  5; 6; 7; 8; 9; 10; 11; 12; 13; 14 ] #
println(N)
mesh_type  = "Quadrangles"
eV_quad    = zeros(size(N))
eP_quad    = zeros(size(N))
eTau_quad  = zeros(size(N))
t_quad     = zeros(size(N))
ndof_quad  = zeros(size(N))
for k=1:length(N)
    t_quad[k]    = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii = main( N[k], mesh_type, order, new )
    eV_quad[k]   = err_V
    eP_quad[k]   = err_P
    eTau_quad[k] = err_Tii
    ndof_quad[k] = ndof
end

mesh_type  = "UnstructTriangles"
eV_tri     = zeros(size(N))
eP_tri     = zeros(size(N))
eTau_tri   = zeros(size(N))
t_tri      = zeros(size(N))
ndof_tri   = zeros(size(N))
for k=1:length(N)
    t_tri[k]     = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii  = main( N[k], mesh_type, order, new )
    eV_tri[k]    = err_V
    eP_tri[k]    = err_P
    eTau_tri[k]  = err_Tii
    ndof_tri[k]  = ndof
end

#################### ORDER 2
order = 2 

mesh_type     = "Quadrangles"
eV_quad_o2    = zeros(size(N))
eP_quad_o2    = zeros(size(N))
eTau_quad_o2  = zeros(size(N))
t_quad_o2     = zeros(size(N))
ndof_quad_o2  = zeros(size(N))
for k=1:length(N)
    t_quad_o2[k]    = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii = main( N[k], mesh_type, order, new )
    eV_quad_o2[k]   = err_V
    eP_quad_o2[k]   = err_P
    eTau_quad_o2[k] = err_Tii
    ndof_quad_o2[k] = ndof
end

mesh_type     = "UnstructTriangles"
eV_tri_o2     = zeros(size(N))
eP_tri_o2     = zeros(size(N))
eTau_tri_o2   = zeros(size(N))
t_tri_o2      = zeros(size(N))
ndof_tri_o2   = zeros(size(N))
for k=1:length(N)
    t_tri_o2[k]     = @elapsed ndof, err_Vx, err_Vy, err_Txx, err_Tyy, err_Txy, err_P, err_V, err_Tii  = main( N[k], mesh_type, order, new )
    eV_tri_o2[k]    = err_V
    eP_tri_o2[k]    = err_P
    eTau_tri_o2[k]  = err_Tii
    ndof_tri_o2[k]  = ndof
end

#######################################

p = Plots.plot(  log10.(1.0 ./ N) , log10.(eV_quad),   markershape=:rect,      color=:blue,                         label="Quads V O1"                          )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_quad),   markershape=:rect,      color=:blue,      linestyle = :dot,  label="Quads P O1"                          )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_quad), markershape=:rect,      color=:blue,      linestyle = :dash, label="Quads Tau O1"                        )
p = Plots.plot( log10.(1.0 ./ N) , log10.(eV_tri),    markershape=:dtriangle, color=:blue,                         label="Triangles V O1"                     )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_tri),    markershape=:dtriangle, color=:blue,      linestyle = :dot,  label="Triangles P O1"                     )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_tri),  markershape=:dtriangle, color=:blue,      linestyle = :dash, label="Triangles Tau O1")#, legend=:bottomright, xlabel = "log_10(h_x)", ylabel = "log_10(err_T)" )

p = Plots.plot!( log10.(1.0 ./ N) , log10.(eV_quad_o2),   markershape=:rect,      color=:red,                         label="Quads V O2"                          )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_quad_o2),   markershape=:rect,      color=:red,      linestyle = :dot,  label="Quads P O2"                          )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_quad_o2), markershape=:rect,      color=:red,      linestyle = :dash, label="Quads Tau O2"                        )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eV_tri_o2),    markershape=:dtriangle, color=:red,                         label="Triangles V O2"                     )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eP_tri_o2),    markershape=:dtriangle, color=:red,      linestyle = :dot,  label="Triangles P O2"                     )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eTau_tri_o2),  markershape=:dtriangle, color=:red,      linestyle = :dash, label="Triangles Tau O2", legend=:outertopright, xlabel = "log_10(h_x)", ylabel = "log_10(err_T)" )
order1 = [2e-4, 1e-4]
order2 = [4e-4, 1e-4]
n      = [35, 70]
p = Plots.plot!( log10.(1.0 ./ n) , log10.(order1), color=:black, label="Order 1")
p = Plots.plot!( log10.(1.0 ./ n) , log10.(order2), color=:black, label="Order 2", linestyle = :dash)
p = Plots.annotate!(log10.(1.0 ./ N[1]), log10(order1[1]), "O1", :black)
p = Plots.annotate!(log10.(1.0 ./ N[1]), log10(order2[1]), "O2", :black, legend=:bottomleft )

# p = Plots.plot(  ndof_quad[2:end], t_quad[2:end], markershape=:rect,      label="Quads"                          )
# p = Plots.plot!( ndof_tri[2:end],  t_tri[2:end],  markershape=:dtriangle, label="Triangles", legend=:bottomright, xlabel = "ndof", ylabel = "time" )
display(p)


# # n = 2
# # tau = 1.0
# # main(n, tau)

# # # n    = collect(1:1:16)
# # # n    = collect(17:1:20) # p2
# # # tau  = collect(4:1:25)
# # n    = collect(1:1:20)
# # tau  = collect(1:1:4)
# # resu = zeros(length(n), length(tau))
# # resp = zeros(length(n), length(tau))

# # for in = 1:length(n)
# #     for it = 1:length(tau)
# #         rp, ru = main(n[in], tau[it])
# #         resu[in,it] = ru
# #         resp[in,it] = rp
# #     end
# # end

# # # p2 = Plots.heatmap(n, tau, resp', c=:jet1 )
# # # display(Plots.plot(p2))

# # file = matopen(string(@__DIR__,"/results/MaxPerr_p3.mat"), "w" )
# # write(file, "n",        n )
# # write(file, "tau",    tau )
# # write(file, "resu",  resu )
# # write(file, "resp",  resp )
# # close(file)