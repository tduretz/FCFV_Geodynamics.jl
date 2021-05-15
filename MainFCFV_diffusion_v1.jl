include("CreateMesh.jl")
using LoopVectorization
using CairoMakie
import AbstractPlotting.GeometryBasics
using SparseArrays, LinearAlgebra
import UnicodePlots 

@views function PlotMakie( Mesh, v )
    p   = [AbstractPlotting.GeometryBasics.Polygon( Point2f0[ (mesh.xv[Mesh.e2v[i,j]], mesh.yv[Mesh.e2v[i,j]]) for j=1:mesh.nf_el] ) for i in 1:Mesh.nel]
    display(poly(p, color = v, colormap = :viridis, strokewidth = 0, strokecolor = :black, markerstrokewidth=0, markerstrokecolor = (0, 0, 0, 0)))
    return 
end

function ComputeCentroids!( mesh )
    aa = 1.0/mesh.nf_el
    @avx for i=1:mesh.nel
        mesh.xc[i] = 0;
        mesh.yc[i] = 0;
        for j=1:mesh.nf_el
            mesh.xc[i] += aa * mesh.xv[mesh.e2v[j,i]] 
            mesh.yc[i] += aa * mesh.yv[mesh.e2v[j,i]] 
        end
    end
end

function Tanalytic2!( mesh, T, a, b, c, d, alp, bet )
    # Evaluate T analytic on barycentres
    @avx for i=1:mesh.nel
        T[i] = exp(alp*sin(a*mesh.xc[i] + c*mesh.yc[i]) + bet*cos(b*mesh.xc[i] + d*mesh.yc[i]))
    end
    return
end

# @views function main()

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = 100, 100
    quad = false

    # FCFV parameter
    tau = 1e-3

    if quad==true 
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax )
    elseif quad==false  
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax ) 
    end

    println("Number of elements: ", mesh.nel)

    # Generate function to be visualised on mesh
    alp = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;

    Tanal  = zeros(mesh.nel)
    ncalls = 4
    # A) Loop version with @avx
    for icall=1:ncalls
        @time Tanalytic2!(mesh , Tanal, a, b, c, d, alp, bet)
    end

    # Compute some mesh vectors 
    se = zeros(mesh.nel)
    ae = zeros(mesh.nel)
    be = zeros(mesh.nel)
    ze = zeros(mesh.nel,2)

    # Dirichlet values on cell faces
    Tdir = zeros(mesh.nf)
    Tneu = zeros(mesh.nf)
    @avx for in=1:mesh.nf
        x        = mesh.xf[in]
        y        = mesh.yf[in]
        Tdir[in] = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
    end

    # Source term
    @avx for iel=1:mesh.nel
        x       = mesh.xc[iel]
        y       = mesh.yc[iel]   
        T       = exp(alp*sin(a*x + c*y) + bet*cos(b*x + d*y))
        se[iel] = T*(-a*alp*cos(a*x + c*y) + b*bet*sin(b*x + d*y))*(a*alp*cos(a*x + c*y) - b*bet*sin(b*x + d*y)) + T*(a^2*alp*sin(a*x + c*y) + b^2*bet*cos(b*x + d*y)) + T*(-alp*c*cos(a*x + c*y) + bet*d*sin(b*x + d*y))*(alp*c*cos(a*x + c*y) - bet*d*sin(b*x + d*y)) + T*(alp*c^2*sin(a*x + c*y) + bet*d^2*cos(b*x + d*y))
    end

    # Assemble FCFV elements
    for iel=1:mesh.nel  

        be[iel] = be[iel]   + mesh.vole[iel]*se[iel]
        
        for ifac=1:mesh.nf_el
            
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = tau#*dAi                              # Stabilisation parameter for the face

            # Assemble
            ze[iel,1] += (bc==1) * dAi*ni_x*Tdir[nodei]  # Dirichlet
            ze[iel,2] += (bc==1) * dAi*ni_y*Tdir[nodei]  # Dirichlet
            be[iel]   += (bc==1) * dAi*taui*Tdir[nodei]                # Dirichlet
            ae[iel]   +=           dAi*taui
            
        end
    end

    rows = Int64[]
    cols = Int64[]
    vals = Float64[]
    f    = zeros(mesh.nf);

    for iel=1:mesh.nel 

        Ke = zeros(4,4);
        fe = zeros(4,1);
        
        for ifac=1:mesh.nf_el 

            nodei  = mesh.e2f[iel,ifac]
            bci    = mesh.bc[nodei]
            
            if bci != 1

                dAi  = mesh.dA[iel,ifac]
                ni_x = mesh.n_x[iel,ifac]
                ni_y = mesh.n_y[iel,ifac]
                taui = tau#*dAi
                
                for jfac=1:mesh.nf_el

                    nodej  = mesh.e2f[iel,jfac]
                    bcj    = mesh.bc[nodej]
                    
                    if bcj!= 1
                        
                        dAj  = mesh.dA[iel,jfac]
                        nj_x = mesh.n_x[iel,jfac]
                        nj_y = mesh.n_y[iel,jfac]
                        tauj = tau#*dAj
                        
                        # Delta
                        del = 0.0
                        if ifac==jfac; del = 1.0; end
                        
                        # Element matrix
                        nitnj         = ni_x*nj_x + ni_y*nj_y;
                        Ke[ifac,jfac] = dAi * (1.0/ae[iel] * dAj * taui*tauj - 1.0/mesh.vole[iel]*dAi*nitnj - taui*del);
                    end
                    
                end

                # RHS vector
                Xi = 0.0;
                if bci == 2; Xi = 1.0; end # indicates Neumann dof
                ti = Tneu[nodei]
                nitze     = ni_x*ze[iel,1] + ni_y*ze[iel,2]
                fe[ifac] += dAi * (1.0/mesh.vole[iel]*nitze - ti*Xi - 1.0/ae[iel]*be[iel]*taui)
                
            end
        end
        # println(Ke)

        for ifac=1:mesh.nf_el

            nodei  = mesh.e2f[iel,ifac]
            bci    = mesh.bc[nodei]
            
            if bci != 1

                for jfac=1:mesh.nf_el

                    nodej  = mesh.e2f[iel,jfac]
                    bcj    = mesh.bc[nodei]

                    if bcj != 1
                        push!(rows, mesh.e2f[ iel,ifac])  
                        push!(cols, mesh.e2f[ iel,jfac]) 
                        push!(vals,       -Ke[ifac,jfac])
                    end
                end
                f[mesh.e2f[ iel,ifac]] -= fe[ifac]
            else
                push!(rows, mesh.e2f[ iel,ifac])  
                push!(cols, mesh.e2f[ iel,ifac]) 
                push!(vals,                 1.0)
                f[mesh.e2f[ iel,ifac]] = Tdir[nodei]
            end
        end

    end
    K = sparse(rows, cols, vals, mesh.nf, mesh.nf)
    droptol!(K, 1e-6)
    uh   = K\f

    # Reconstruct element values
    ue          = zeros(mesh.nel);
    qe          = zeros(mesh.nel,2);
    for iel=1:mesh.nel
    
        ue[iel]   = be[iel]/ae[iel]
        qe[iel,1] =-1.0/mesh.vole[iel]*ze[iel,1]
        qe[iel,2] =-1.0/mesh.vole[iel]*ze[iel,2]
        
        for ifac=1:mesh.nf_el
            
            # Face
            nodei = mesh.e2f[iel,ifac]
            bc    = mesh.bc[nodei]
            dAi   = mesh.dA[iel,ifac]
            ni_x  = mesh.n_x[iel,ifac]
            ni_y  = mesh.n_y[iel,ifac]
            taui  = tau#*dAi    # Stabilisation parameter for the face

            # Assemble
            ue[iel]   += (bc!=1) *  dAi*taui*uh[mesh.e2f[iel, ifac]]/ae[iel]
            qe[iel,1] -= (bc!=1) *  1.0/mesh.vole[iel]*dAi*ni_x*uh[mesh.e2f[iel, ifac]]
            qe[iel,2] -= (bc!=1) *  1.0/mesh.vole[iel]*dAi*ni_y*uh[mesh.e2f[iel, ifac]]
         end
    end
    
    # display(UnicodePlots.spy(K))
    # println(uh[:])

    # for iel=1:mesh.nel
    #     println(ue[iel])
    # end

    # for i=1:length(uh)
    #     println(uh[i])
    # end

    # println(mesh.e2f)


    import MAT
    file   = MAT.matopen("/Users/imac/ownCloud/FCFV/Mat100_ue.mat")
    ue_mat = MAT.read(file, "ue")

    # Visualise
    @time PlotMakie( mesh, ue )
    # function qplot(x, y, v)
    # xc=LinRange(xmin,xmax,nx)
    # yc=LinRange(ymin,ymax,ny)
    # display( heatmap(xc,yc,reshape(ue,nx,ny), clim=[0.7 1.5]) )
        # clf()
        # display( pcolor(xc, yc, reshape(ue,nx,ny)) )
        # colorbar()
        # xlabel("x")
        # ylabel("y")
        # show()
    # end

# end

# main()