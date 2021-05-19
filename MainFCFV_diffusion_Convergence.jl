include("CreateMeshFCFV.jl")
include("VisuFCFV.jl")
include("DiscretisationFCFV.jl")
using LoopVectorization
using SparseArrays, LinearAlgebra
import UnicodePlots 
import Plots

function SetUpProblem!( mesh, T, Tdir, Tneu, se, a, b, c, d, alp, bet )
    # Evaluate T analytic on cell faces
    @avx for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        Tdir[in] = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
    end
    # Evaluate T analytic on barycentres
    @avx for iel=1:mesh.nel
        x       = mesh.xc[iel]
        y       = mesh.yc[iel]
        T       = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        T[iel]  = T
        se[iel] = T*(-a*alp*cos(a*x + c*y) + b*bet*sin(b*x + d*y))*(a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y)) + T*(a^2*alp*sin(a*x + c*y) + b^2*bet*cos(b*x + d*y)) + T*(-alp*c*cos(a*x + c*y) + bet*d*sin(b*x + d*y))*(alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y)) + T*(alp*c^2*sin(a*x + c*y) + bet*d^2*cos(b*x + d*y))
    end
    return
end

function ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    eT  = zeros(mesh.nel)
    eqx = zeros(mesh.nel)
    eqy = zeros(mesh.nel)
    Ta  = zeros(mesh.nel)
    qxa = zeros(mesh.nel)
    qya = zeros(mesh.nel)
    @avx for iel=1:mesh.nel
        x        = mesh.xc[iel]
        y        = mesh.yc[iel]
        Ta[iel]       = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        qxa[iel]      = -Ta[iel] * (a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y))
        qya[iel]      = -Ta[iel] * (alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y))
        eT[iel]  = Te[iel] - Ta[iel]
        eqx[iel] = qx[iel] - qxa[iel]
        eqy[iel] = qy[iel] - qya[iel]
    end
    errT  = norm(eT)/norm(Ta)
    errqx = norm(eqx)/norm(qxa)
    errqy = norm(eqy)/norm(qya)
    return errT, errqx, errqy
end
    

function StabParam(tau, dA, Vol, mesh_type)
    if mesh_type=="Quadrangles";        taui = tau;   end
    # if mesh_type=="UnstructTriangles";  taui = tau*dA end
    if mesh_type=="UnstructTriangles";  taui = tau end
    return taui
end
    
@views function main( N, mesh_type )  

    println("\n******** FCFV POISSON********")

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = N, N
  
    # Generate mesh
    println("Mesh generation :")
    if mesh_type=="Quadrangles" 
        tau  = 1
        @time mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax )
    elseif mesh_type=="UnstructTriangles"  
        tau  = 1
        @time  mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax ) 
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
    @time ae, be, ze = ComputeFCFV(mesh, se, Tdir, tau)

    # Assemble element matrices and RHS
    println("Compute element matrices:")
    @time Kv, fv = ElementAssemblyLoop(mesh, ae, be, ze, Tdir, Tneu, tau)

    # Assemble triplets and sparse
    println("Assemble triplets and sparse:")
    @time K, f = CreateTripletsSparse(mesh, Kv, fv)
    # display(UnicodePlots.spy(K))

    # Solve for hybrid variable
    println("Direct solve:")
    @time Th   = K\f
    # @time Th   = cholesky(K)\f

    # Reconstruct element values
    println("Compute element values:")
    @time Te, qx, qy = ComputeElementValues(mesh, Th, ae, be, ze, Tdir, tau)

    # Compute discretisation errors
    err_T, err_qx, err_qy = ComputeError( mesh, Te, qx, qy, a, b, c, d, alp, bet )
    println("Error in T:  ", err_T )
    println("Error in qx: ", err_qx)
    println("Error in qy: ", err_qy)

    # Visualise
    # println("Visualisation:")
    # @time PlotMakie( mesh, Te )
    # PlotElements( mesh )
    return err_T, err_qx, err_qy
end

N          = [8, 16, 32, 64, 128, 256, 512]
mesh_type  = "Quadrangles"
eT_quad    = zeros(size(N))
eqx_quad   = zeros(size(N))
eqy_quad   = zeros(size(N))
for k=1:length(N)
    @time err_T, err_qx, err_qy = main( N[k], mesh_type )
    eT_quad[k]  = err_T
    eqx_quad[k] = err_qx
    eqy_quad[k] = err_qy
end

mesh_type  = "UnstructTriangles"
eT_tri     = zeros(size(N))
eqx_tri    = zeros(size(N))
eqy_tri    = zeros(size(N))
for k=1:length(N)
    @time err_T, err_qx, err_qy = main( N[k], mesh_type )
    eT_tri[k]  = err_T
    eqx_tri[k] = err_qx
    eqy_tri[k] = err_qy
end

p = Plots.plot(  log10.(1.0 ./ N) , log10.(eT_quad), markershape=:rect,      label="Quads"                          )
p = Plots.plot!( log10.(1.0 ./ N) , log10.(eT_tri),  markershape=:dtriangle, label="Triangles", legend=:bottomright, xlabel = "log_10(h_x)", ylabel = "log_10(err_T)" )
display(p)