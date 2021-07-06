import TriangleMesh, UnicodePlots, Plots
using Printf, LoopVectorization, LinearAlgebra, SparseArrays

include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV.jl")
include("DiscretisationFCFV_Poisson_o2.jl")
include("SolversFCFV_Poisson.jl")

#--------------------------------------------------------------------#

function SetUpProblem!( mesh, T, Tdir, Tneu, se, a, b, c, d, alp, bet )
    # Evaluate T analytic on cell faces
    @tturbo for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        Tdir[in] = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        dTdx     = Tdir[in] * (a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y))
        dTdy     = Tdir[in] * (alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y))
        Tneu[in] = -dTdy # nx*dTdx + nt*dTdy on SOUTH face
    end
    # Evaluate T analytic on barycentres
    @tturbo for iel=1:mesh.nel
        x       = mesh.xc[iel]
        y       = mesh.yc[iel]
        T       = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        T[iel]  = T
        se[iel] = T*(-a*alp*cos(a*x + c*y) + b*bet*sin(b*x + d*y))*(a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y)) + T*(a^2*alp*sin(a*x + c*y) + b^2*bet*cos(b*x + d*y)) + T*(-alp*c*cos(a*x + c*y) + bet*d*sin(b*x + d*y))*(alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y)) + T*(alp*c^2*sin(a*x + c*y) + bet*d^2*cos(b*x + d*y))
    end
    return
end

#--------------------------------------------------------------------#

function ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    eT  = zeros(mesh.nel)
    eqx = zeros(mesh.nel)
    eqy = zeros(mesh.nel)
    eq  = zeros(mesh.nel)
    Ta  = zeros(mesh.nel)
    qxa = zeros(mesh.nel)
    qya = zeros(mesh.nel)
    qa  = zeros(mesh.nel)
    @tturbo for iel=1:mesh.nel
        x        = mesh.xc[iel]
        y        = mesh.yc[iel]
        Ta[iel]  = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        qxa[iel] = -Ta[iel] * (a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y))
        qya[iel] = -Ta[iel] * (alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y))
        qa[iel]  = sqrt(qxa[iel]^2 + qya[iel]^2)
        q        = sqrt( qx[iel]^2 +  qy[iel]^2)
        eT[iel]  = Te[iel] - Ta[iel]
        eqx[iel] = qx[iel] - qxa[iel]
        eqy[iel] = qy[iel] - qya[iel]
        eq[iel]  = q       - qa[iel]
    end
    errT  = norm(eT) /norm(Ta)
    errqx = norm(eqx)/norm(qxa)
    errqy = norm(eqy)/norm(qya)
    errq  = norm(eq) /norm(qa)
    return errT, errqx, errqy, errq
end
    
#--------------------------------------------------------------------#

function StabParam(tau, dA, Vol, mesh_type)
    if mesh_type=="Quadrangles";        taui = tau;    end
    # if mesh_type=="UnstructTriangles";  taui = tau*dA; end
    if mesh_type=="UnstructTriangles";  taui = tau end
    return taui
end

#--------------------------------------------------------------------#
    
@views function main( N, mesh_type, order )

    println("\n******** FCFV POISSON ********")

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = N, N
    R          = 0.5
    inclusion  = 0
    solver     = 0
    o2         = order-1
    BC         = [2; 1; 1; 1] # S E N W --- 1: Dirichlet / 2: Neumann
    # mesh_type  = "Quadrangles"
    # mesh_type  = "UnstructTriangles"
  
    # Generate mesh
    if mesh_type=="Quadrangles" 
        if o2==0 tau  = 1e0 end
        if o2==1 tau  = 1e4 end
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC )
    elseif mesh_type=="UnstructTriangles"  
        if o2==0 tau  = 8e0 end
        if o2==1 tau  = 1e6 end
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC ) 
    end
    println("Number of elements: ", mesh.nel)

    # Source term and BCs etc...
    Tanal  = zeros(mesh.nel)
    se     = zeros(mesh.nel)
    Tdir   = zeros(mesh.nf)
    Tneu   = zeros(mesh.nf)
    alp = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;
    println("Model configuration :")
    @time SetUpProblem!(mesh , Tanal, Tdir, Tneu, se, a, b, c, d, alp, bet)

    # Compute some mesh vectors 
    println("Compute FCFV vectors:")
    @time ae, be, be_o2, ze, pe, mei, pe, rj  = ComputeFCFV_o2(mesh, se, Tdir, tau, o2)

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    @time Kv, fv = ElementAssemblyLoop_o2(mesh, ae, be, be_o2, ze, mei, pe, rj, Tdir, Tneu, tau, o2)

    # Assemble triplets and sparse
    println("Assemble triplets and sparse:")
    @time K, f = CreateTripletsSparse(mesh, Kv, fv)
    
    # Solve
    println("Solve:")
    @time Th = SolvePoisson(mesh, K, f, solver)

    # Reconstruct element values
    println("Compute element values:")
    @time Te, qx, qy = ComputeElementValues_o2(mesh, Th, ae, be, be_o2, ze, rj, mei, Tdir, tau, o2)

    # Compute discretisation errors
    err_T, err_qx, err_qy, err_q = ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    println("Error in T:  ", err_T )
    println("Error in q:  ", err_q)
    println("Error in qx: ", err_qx)
    println("Error in qy: ", err_qy)

    # # Visualise
    # println("Visualisation:")
    # @time PlotMakie( mesh, Te, xmin, xmax, ymin, ymax, :viridis )

    return mesh.nf, err_T, err_qx, err_qy, err_q
end

#################### ORDER 1
order = 1 

N          = [8, 16, 32, 64 ]#, 128, 256, 512, 1024] 
mesh_type  = "Quadrangles"
eT_quad    = zeros(size(N))
eqx_quad   = zeros(size(N))
eqy_quad   = zeros(size(N))
eq_quad    = zeros(size(N))
t_quad     = zeros(size(N))
ndof_quad  = zeros(size(N))
for k=1:length(N)
    t_quad[k]    = @elapsed ndof, err_T, err_qx, err_qy, err_q = main( N[k], mesh_type, order )
    eT_quad[k]   = err_T
    eqx_quad[k]  = err_qx
    eqy_quad[k]  = err_qy
    eq_quad[k]   = err_q
    ndof_quad[k] = ndof
end

mesh_type  = "UnstructTriangles"
eT_tri     = zeros(size(N))
eqx_tri    = zeros(size(N))
eqy_tri    = zeros(size(N))
eq_tri     = zeros(size(N))
t_tri      = zeros(size(N))
ndof_tri   = zeros(size(N))
for k=1:length(N)
    t_tri[k]     = @elapsed ndof, err_T, err_qx, err_qy, err_q  = main( N[k], mesh_type, order  )
    eT_tri[k]    = err_T
    eqx_tri[k]   = err_qx
    eqy_tri[k]   = err_qy
    eq_tri[k]    = err_q
    ndof_tri[k]  = ndof
end

#################### ORDER 2
order = 2 

N            = [8, 16, 32, 64 ]#, 128, 256, 512, 1024] 
mesh_type    = "Quadrangles"
eT_quad_o2   = zeros(size(N))
eq_quad_o2   = zeros(size(N))
t_quad_o2    = zeros(size(N))
ndof_quad_o2 = zeros(size(N))
for k=1:length(N)
    t_quad_o2[k]    = @elapsed ndof, err_T, err_qx, err_qy, err_q = main( N[k], mesh_type, order )
    eT_quad_o2[k]   = err_T
    eq_quad_o2[k]   = err_q
    ndof_quad_o2[k] = ndof
end

mesh_type    = "UnstructTriangles"
eT_tri_o2    = zeros(size(N))
eq_tri_o2    = zeros(size(N))
t_tri_o2     = zeros(size(N))
ndof_tri_o2  = zeros(size(N))
for k=1:length(N)
    t_tri_o2[k]     = @elapsed ndof, err_T, err_qx, err_qy, err_q  = main( N[k], mesh_type, order  )
    eT_tri_o2[k]    = err_T
    eq_tri_o2[k]    = err_q
    ndof_tri_o2[k]  = ndof
end

#######################################

p = Plots.plot(  log10.(1.0 ./ N) , log10.(eT_quad), markershape=:rect, color=:blue,      label="Quads O1 u"        )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eq_quad), markershape=:rect, linestyle = :dot, color=:blue,     label="Quads O1 q"        )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eT_tri),  markershape=:dtriangle, color=:blue, label="Triangles O1 u"    )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eq_tri),  markershape=:dtriangle, linestyle = :dot, color=:blue, label="Triangles O1 q"    )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eT_quad_o2), markershape=:rect, color=:red,      label="Quads O2 u"     )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eq_quad_o2), markershape=:rect, color=:red, linestyle = :dot,     label="Quads O2 q"     )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eT_tri_o2),  markershape=:dtriangle, color=:red, label="Triangles O2 u" )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eq_tri_o2),  markershape=:dtriangle, color=:red, linestyle = :dot, label="Triangles O2 q", legend=:bottomright, xlabel = "log_10(h_x)", ylabel = "log_10(err_T)" )
# p = Plots.plot(  ndof_quad[2:end], t_quad[2:end], markershape=:rect,      label="Quads"                          )
# p = Plots.plot!( ndof_tri[2:end],  t_tri[2:end],  markershape=:dtriangle, label="Triangles", legend=:bottomright, xlabel = "ndof", ylabel = "time" )
display(p)