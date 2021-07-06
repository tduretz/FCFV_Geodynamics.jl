import TriangleMesh

Base.@kwdef mutable struct FCFV_Mesh
    type   ::Union{String, Missing}          = missing
    nel    ::Union{Int64,  Missing}          = missing
    nf     ::Union{Int64,  Missing}          = missing
    nv     ::Union{Int64,  Missing}          = missing
    nn_el  ::Union{Int64,  Missing}          = missing
    nf_el  ::Union{Int64,  Missing}          = missing
    xv     ::Union{Vector{Float64}, Missing} = missing # node x coordinate
    yv     ::Union{Vector{Float64}, Missing} = missing # node y coordinate
    xf     ::Union{Vector{Float64}, Missing} = missing # face x coordinate
    yf     ::Union{Vector{Float64}, Missing} = missing # face y coordinate
    bc     ::Union{Vector{Int64},   Missing} = missing # face y coordinate
    xc     ::Union{Vector{Float64}, Missing} = missing # cent x coordinate
    yc     ::Union{Vector{Float64}, Missing} = missing # cent y coordinate
    e2v    ::Union{Matrix{Int64},   Missing} = missing # cell 2 node numbering
    e2f    ::Union{Matrix{Int64},   Missing} = missing # cell 2 face numbering
    vole   ::Union{Vector{Float64}, Missing} = missing # volume of element
    n_x    ::Union{Matrix{Float64}, Missing} = missing # normal 2 face x
    n_y    ::Union{Matrix{Float64}, Missing} = missing # normal 2 face y
    dA     ::Union{Matrix{Float64}, Missing} = missing # face length
    e2e    ::Union{Matrix{Int64},   Missing} = missing # element 2 element numbering
    f2e    ::Union{Matrix{Int64},   Missing} = missing # face 2 element numbering
    dA_f   ::Union{Matrix{Float64}, Missing} = missing # face 2 element numbering
    vole_f ::Union{Matrix{Float64}, Missing} = missing # volume of element
    n_x_f  ::Union{Matrix{Float64}, Missing} = missing # normal 2 face x
    n_y_f  ::Union{Matrix{Float64}, Missing} = missing # normal 2 face y
    # ---- mat props ---- #
    ke     ::Union{Vector{Float64}, Missing} = missing # diffusion coefficient
    phase  ::Union{Vector{Float64}, Missing} = missing # phase
    # phase_f::Union{Matrix{Float64}, Missing} = missing # normal 2 face x
end

#--------------------------------------------------------------------#

function MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC=[2; 1; 1; 1;], area = ((xmax-xmin)/nx)*((ymax-ymin)/ny), no_pts_incl = Int64(floor(1.0*pi*R/sqrt(((xmax-xmin)/nx)^2+((ymax-ymin)/ny)^2)))  )

    regions  = Array{Float64}(undef,4,0)
    holes    = Array{Float64}(undef,2,0)
    dx       = (xmax-xmin)/nx
    dy       = (ymax-ymin)/ny
    pts_l    = 1;
    pts_u    = 0;
    
    # 1. Four corners of the domain
    px     = [xmin; xmax; xmax; xmin]
    py     = [ymin; ymin; ymax; ymax]
    sx     = [ 1; 2; 3; 4; ] 
    sy     = [ 2; 3; 4; 1; ]
    st     = BC  # segment markers == boundary flags
    
    println(sx)
    println(st)
    
    no_pts = size(px,1);
    pts_l  = pts_l+no_pts;
    pts_u  = pts_u+no_pts;
    # Region 1
    h1     = [xmin+1e-13; ymin+1e-13; 1.0; 0.0] 

    if inclusion==1
        # 2. Inclusion
        theta0       = collect(LinRange(0,2*pi,no_pts_incl+1));
        theta        = theta0[1:end-1] # do not include last point (periodic 0 == 2pi)
        xx           = cos.(theta);
        yy           = sin.(theta);
        center_x     = (xmax + xmin)/2.0
        center_y     = (ymax + ymin)/2.0
        X            = center_x .+ R*xx;
        Y            = center_y .+ R*yy;
        no_pts       = length(X);
        st1          = 3*ones(1,no_pts);
        pts_u        = pts_u + no_pts;
        sx1          = collect(pts_l:pts_u)
        sy1          = collect(pts_l+1:pts_u+1)
        sy1[end]     = pts_l # Periodic
        h2           = [0.0; 0.0; 2.0; 0.0] # Region 2
        # println(X.^2 .+ Y.^2)
        # println(X)
        # println(Y)
        # println(sx1)
        # println(sy1)
        # println(st1)
        for i=1:no_pts_incl
            px   = push!(px, X[i])
            py   = push!(py, Y[i])
            sx   = push!(sx, sx1[i])
            sy   = push!(sy, sy1[i])
            st   = push!(st, st1[i])
        end
        regions = hcat(h1,h2)
    end

    p       = hcat(px, py)         # points
    s       = hcat(sx, sy)         # segments
    p       = p'
    s       = s'
    st      = st[:]
    
    # Triangulation
    domain   = TriangleMesh.Polygon_pslg(size(p,2), p, 0, Array{Int64}(undef,2,0), 0, Array{Float64}(undef,2,0),  size(s,2), s, st, 0, holes, size(regions,2), regions, 1.0)
    astring  = @sprintf("%0.10lf", area)
    switches = "vQDpenq33o2IAa$(astring)"  #QDpeq33o2Aa0.01
    
    println("Arguments to Triangle: ", switches)
    trimesh  = TriangleMesh.create_mesh(domain, switches)
    nvert_el = 3 # vertices per element
    
    mesh        = FCFV_Mesh()
    mesh.type   = "UnstructTriangles"
    mesh.nel    = trimesh.n_cell
    e2v         = trimesh.cell[1:3,:]
    mesh.nv     = maximum(e2v)
    e2f         = trimesh.cell[4:6,:] .- mesh.nv
    mesh.nf     = maximum(e2f)
    mesh.xv     = trimesh.point[1,1:mesh.nv]
    mesh.yv     = trimesh.point[2,1:mesh.nv]
    mesh.xf     = trimesh.point[1,mesh.nv+1:end]
    mesh.yf     = trimesh.point[2,mesh.nv+1:end]
    mesh.bc     = trimesh.point_marker[mesh.nv+1:end]
    mesh.ke     = ones(Float64,mesh.nel)

    if length(trimesh.triangle_attribute)>0
        println(minimum(trimesh.triangle_attribute))
        println(maximum(trimesh.triangle_attribute))
        mesh.phase  = trimesh.triangle_attribute
    else
        mesh.phase  = ones(trimesh.n_cell)
    end
    
    nel  = trimesh.n_cell
    vole = zeros(nel)
    xc   = zeros(nel)
    yc   = zeros(nel)
    
    @tturbo for iel=1:nel
        # Compute volumes of triangles - use vertices coordinates
        x1 = mesh.xv[e2v[1,iel]]
        y1 = mesh.yv[e2v[1,iel]]
        x2 = mesh.xv[e2v[2,iel]]
        y2 = mesh.yv[e2v[2,iel]]
        x3 = mesh.xv[e2v[3,iel]]
        y3 = mesh.yv[e2v[3,iel]]
        a         = sqrt((x1-x2)^2 + (y1-y2)^2)
        b         = sqrt((x2-x3)^2 + (y2-y3)^2)
        c         = sqrt((x1-x3)^2 + (y1-y3)^2)
        s         = 1/2*(a+b+c)
        vole[iel] = sqrt(s*(s-a)*(s-b)*(s-c))
        xc[iel]   = 1.0/3.0*(x1+x2+x3)
        yc[iel]   = 1.0/3.0*(y1+y2+y3)
    end
    
    mesh.e2v    = e2v'
    mesh.e2f    = e2f'
    mesh.nn_el  = 3
    mesh.nf_el  = 3
    mesh.xc     = xc
    mesh.yc     = yc
    mesh.vole   = vole
    
    nodeA = [2 3 1]
    nodeB = [3 1 2]
    nodeC = [1 2 3]
    
    # Compute normal to faces
    mesh.n_x = zeros(Float64,mesh.nel,mesh.nf_el)
    mesh.n_y = zeros(Float64,mesh.nel,mesh.nf_el)
    mesh.dA  = zeros(Float64,mesh.nel,mesh.nf_el)
    mesh.e2e = zeros(  Int64,mesh.nel,mesh.nf_el)
    mesh.ke  =  ones(Float64,mesh.nel)
    
     # Assemble FCFV elements
     @tturbo for iel=1:mesh.nel 
        
        for ifac=1:mesh.nf_el
            
            nodei  = mesh.e2f[iel,ifac]

            # Neighbouring element for this face
            ele1 = trimesh.voronoi.edge[1,nodei]
            ele2 = trimesh.voronoi.edge[2,nodei]
            iel2 = (iel==ele1) * ele2 + (iel==ele2) * ele1;
            mesh.e2e[iel,ifac] = iel2;

            # println(" element: ", iel, " connects: ", iel2)
            
            # Vertices
            vert1  = mesh.e2v[iel,nodeA[ifac]]
            vert2  = mesh.e2v[iel,nodeB[ifac]]
            vert3  = mesh.e2v[iel,nodeC[ifac]]
            bc     = mesh.bc[nodei]
            dx     = (mesh.xv[vert1] - mesh.xv[vert2] );
            dy     = (mesh.yv[vert1] - mesh.yv[vert2] );
            dAi    = sqrt(dx^2 + dy^2);
    
            # Face normal
            n_x  = -dy/dAi
            n_y  =  dx/dAi
            
            # Third vector
            v_x  = mesh.xf[nodei] - mesh.xc[iel]
            v_y  = mesh.yf[nodei] - mesh.yc[iel]
            
            # Check wether the normal points outwards
            dot                 = n_x*v_x + n_y*v_y 
            mesh.n_x[iel,ifac]  = ((dot>=0.0)*n_x - (dot<0.0)*n_x)
            mesh.n_y[iel,ifac]  = ((dot>=0.0)*n_y - (dot<0.0)*n_y)
            mesh.dA[iel,ifac]   = dAi
        end
    end

    ############# FOR THE PSEUDO-TRANSIENT PURPOSES ONLY #############

    # Create face to element numbering
    mesh.f2e    = zeros(Int,mesh.nf,2)
    mesh.vole_f = zeros(Float64,mesh.nf,2)
    mesh.n_x_f  = zeros(Float64,mesh.nf,2)
    mesh.n_y_f  = zeros(Float64,mesh.nf,2)
    mesh.dA_f   = zeros(Float64,mesh.nf,2)

    # # Loop through field names and fields: standard
    # for fname in fieldnames(typeof(trimesh))
    #     println("Field name: ", fname)
    #     println("Content   : ", getfield(trimesh, fname))
    # end

    # Loop over edges and uses Voronoï diagram to get adjacent cells
    @tturbo for ifac=1:mesh.nf 
        mesh.f2e[ifac,1] = trimesh.voronoi.edge[1,ifac]
        mesh.f2e[ifac,2] = trimesh.voronoi.edge[2,ifac]
        act1 = trimesh.voronoi.edge[1,ifac] > 0
        act2 = trimesh.voronoi.edge[2,ifac] > 0
        iel1 = (act1==1) * trimesh.voronoi.edge[1,ifac] + (act1==0) * 1
        iel2 = (act2==1) * trimesh.voronoi.edge[2,ifac] + (act2==0) * 1
        # Compute face length
        vert1  = trimesh.edge[1,ifac]
        vert2  = trimesh.edge[2,ifac]
        xf     = 0.5*(mesh.xv[vert1] + mesh.xv[vert2])
        yf     = 0.5*(mesh.yv[vert1] + mesh.yv[vert2])
        dx     = (mesh.xv[vert1] - mesh.xv[vert2] );
        dy     = (mesh.yv[vert1] - mesh.yv[vert2] );
        dAi    = sqrt(dx^2 + dy^2);    
        mesh.dA_f[ifac,1] =  dAi
        mesh.dA_f[ifac,2] =  dAi
        # Volumes
        mesh.vole_f[ifac,1] = (act1==1) * mesh.vole[iel1]
        mesh.vole_f[ifac,2] = (act2==2) * mesh.vole[iel2]
        # Compute face normal
        n_x  = -dy/dAi
        n_y  =  dx/dAi
        # Third vector  (w.r.t. element 1)
        v_x  = xf - mesh.xc[iel1]
        v_y  = yf - mesh.yc[iel1]
        # Check wether the normal points outwards
        dot                 = n_x*v_x + n_y*v_y 
        mesh.n_x_f[ifac,1]  = (act1==1) * ((dot>=0.0)*n_x - (dot<0.0)*n_x)
        mesh.n_y_f[ifac,1]  = (act1==1) * ((dot>=0.0)*n_y - (dot<0.0)*n_y)
        # Take the negative for the second element
        mesh.n_x_f[ifac,2]  = -mesh.n_x_f[ifac,1]
        mesh.n_y_f[ifac,2]  = -mesh.n_y_f[ifac,1]
    end
    ############# FOR THE PSEUDO-TRANSIENT PURPOSES ONLY #############
    
    return mesh
    
    end

#--------------------------------------------------------------------#

function MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax, inclusion, R, BC=[2; 1; 1; 1;] )
    # Structure
    mesh        = FCFV_Mesh()
    # Generates a 2D rectangular mesh of nx*ny cells
    ncell      = nx*ny
    # Twice resolution mesh
    nx2 = 2*nx + 1
    ny2 = 2*ny + 1
    # 1D axis
    dx,dy = (xmax-xmin)/nx, (xmax-xmin)/ny
    xc  = collect(LinRange(xmin+dx/2.0,xmax-dx/2.0,nx))
    yc  = collect(LinRange(ymin+dy/2.0,ymax-dy/2.0,ny))
    x2  = LinRange(xmin,xmax,nx2)
    y2  = LinRange(ymin,ymax,ny2)
    # 2D mesh: fake ngrid for P-T space ;)
    x2d   = repeat(x2, 1, length(y2))'
    y2d   = repeat(y2, 1, length(x2))
    nodes = zeros(Int64, size(x2d))
    # Number active dofs (midfaces) - Face node numbering
    inum = 0
    for i=1:nx2
        for j=1:ny2
            if (mod(j,2)==1 && mod(i,2)==0) || (mod(j,2)==0 && mod(i,2)==1)
                inum += 1
                nodes[i,j] = inum
            end
        end
    end
    nface = inum;
    # Number vertices
    inum = 0
    for i=1:nx2
        for j=1:ny2
            if (mod(j,2)==1 && mod(i,2)==1) || (mod(j,2)==1 && mod(i,2)==1)
                inum += 1
                nodes[i,j] = inum
            end
        end
    end
    # Face nodes
    xf = zeros(nface) 
    yf = zeros(nface)
    tf = zeros(Int64,nface) # node type - 1 boundary
    for i=1:nx2
        for j=1:ny2
            if (mod(j,2)==1 && mod(i,2)==0) || (mod(j,2)==0 && mod(i,2)==1)
                xf[nodes[i,j]] = x2d[i,j]
                yf[nodes[i,j]] = y2d[i,j]
                if ( j==1 ) # WEST
                    tf[nodes[i,j]] = BC[4]
                end
                if ( j==ny2 ) # EAST
                    tf[nodes[i,j]] = BC[2]
                end
                if ( i==nx2 ) # NORTH
                    tf[nodes[i,j]] = BC[3]
                end
                if ( i==1 ) # SOUTH
                    tf[nodes[i,j]] = BC[1] # set Neumann at the South
                end
            end
        end
    end
    # Vertices nodes
    xn = zeros( (nx+1)*(ny+1) ) 
    yn = zeros( (nx+1)*(ny+1) )
    tn = zeros(Int64, (nx+1)*(ny+1) ) # node type - 1 boundary
    for i=1:nx+1
        for j=1:ny+1
            ii = i + (j-1)*(ny+1)
                xn[ii] = xmin + (i-1)*dx
                yn[ii] = ymin + (j-1)*dy
                if (i==1 || i==nx+1 || j==1 || j==ny+1)
                    tn[ii] = 1
                end
        end
    end
    # Cell 2 face numbering - for matrix connectivity
    face = zeros(Int64, 4, ncell)
    e2e  = zeros(Int64, 4, ncell)
    @tturbo for i=1:nx
        for j=1:ny
            k  = j + (i-1)*ny
            jc = i+1 + (i-1)*1
            ic = j+1 + (j-1)*1
            face[1,k] =  nodes[ic,jc-1] # South
            e2e[1,k]  =  ((i-1) > 0) * (j + (i-1-1)*ny)  + ((i-1) <= 0) * -1
            
            face[2,k] =  nodes[ic-1,jc] # West
            e2e[2,k]  =  (j-1 > 0) * (j-1 + (i-1)*ny)  + ((j-1) <= 0) * -1

            face[3,k] =  nodes[ic+1,jc] # East
            e2e[3,k]  =  (j+1 <= ny ) * (j+1 + (i-1)*ny)  + (j+1 > ny) * -1

            face[4,k] =  nodes[ic,jc+1] # North
            e2e[4,k]  =  (i+1 <=nx) * (j + (i-1+1)*ny) + (i+1 >nx) * -1
        end
    end

    # Cell 2 vertices - used for visualisation of quads
    vert = zeros(Int64, 4, ncell)
    @tturbo for i=1:nx
        for j=1:ny
            k  = j + (i-1)*ny
            jc = i+1 + (i-1)*1
            ic = j+1 + (j-1)*1
            vert[1,k] =  nodes[ic-1,jc+1] # South
            vert[2,k] =  nodes[ic-1,jc-1] # East
            vert[3,k] =  nodes[ic+1,jc-1] # North
            vert[4,k] =  nodes[ic+1,jc+1] # West
        end
    end
    # Centroids
    xc = zeros(ncell)
    yc = zeros(ncell)
    w  = 0.25
    @tturbo for iel=1:ncell
        tempx = 0
        tempy = 0
        for j=1:4
            tempx += w * xf[face[j,iel]]
            tempy += w * yf[face[j,iel]]
        end
        xc[iel] = tempx
        yc[iel] = tempy
    end

    # Fill structure
    mesh.type   = "Quadrangles"
    mesh.nel    = ncell
    mesh.nf     = nface
    mesh.nf_el  = 4
    mesh.nf_el  = 4
    mesh.nv     = (nx+1)*(ny+1)
    mesh.xv     = xn
    mesh.yv     = yn
    mesh.xf     = xf
    mesh.yf     = yf
    mesh.e2v    = vert' # cell nodes - not dofs! Dofs are in 'cell'
    mesh.e2f    = face'
    mesh.xc     = xc
    mesh.yc     = yc
    mesh.vole   = dx*dy*ones(ncell)
    mesh.bc     = tf
    mesh.ke     =  ones(Float64,mesh.nel)
    mesh.e2e    = e2e'

    nodeA = [2 1 3 4]
    nodeB = [3 2 4 1]

    # phase
    mesh.phase = ones(mesh.nel)
    if inclusion==1 
        @tturbo for iel=1:mesh.nel
            x               = mesh.xc[iel]
            y               = mesh.yc[iel]
            out             = (x^2 + y^2)>R^2
            mesh.phase[iel] = (out==1)*1.0 + (out!=1)*2.0
        end
    end

    # Compute normal to faces
    mesh.n_x = zeros(Float64,mesh.nel,mesh.nf_el)
    mesh.n_y = zeros(Float64,mesh.nel,mesh.nf_el)
    mesh.dA  = zeros(Float64,mesh.nel,mesh.nf_el)

    # Get the number of element that are on each side of a face
    face2element   = zeros(Int64,2,mesh.nf)
    face2vertices  = zeros(Int64,2,mesh.nf)
    elem           = collect(1:mesh.nel)
    @tturbo for iel=1:mesh.nel 
        for ifac=1:mesh.nf_el
            nodei  = mesh.e2f[iel,ifac]
            vert1  = mesh.e2v[iel,nodeA[ifac]]
            vert2  = mesh.e2v[iel,nodeB[ifac]]
            # println("element: ",  iel, " neighbour: ", mesh.e2e[iel,ifac])
            # if mesh.e2e[iel,ifac] == -1 n0+=1 end
            face2element[1,nodei]  = elem[iel]
            face2element[2,nodei]  = mesh.e2e[iel,ifac]
            face2vertices[1,nodei] = vert1
            face2vertices[2,nodei] = vert2
        end
    end
    
    # Flag interface faces
    for iel=1:mesh.nel 
        type1 = Int64(mesh.phase[iel])
        for ifac=1:mesh.nf_el
            nodei = mesh.e2f[iel,ifac]
            iel2  = mesh.e2e[iel,ifac]
            ind   = (iel2>0) * iel2 + (iel2<0) * iel 
            type2 = Int64(mesh.phase[ind]) 
            flag  = mesh.bc[nodei]
            yes   = (type1 != type2)
            mesh.bc[nodei] = (yes==0) * flag + (yes==1) * 3
        end
    end

     # Assemble FCFV elements
    @tturbo for iel=1:mesh.nel  
        
        # println("element: ",  iel)

        for ifac=1:mesh.nf_el
            
            nodei  = mesh.e2f[iel,ifac]

            # Vertices
            vert1  = mesh.e2v[iel,nodeA[ifac]]
            vert2  = mesh.e2v[iel,nodeB[ifac]]
            bc     = mesh.bc[nodei]
            dx     = abs(mesh.xv[vert1] - mesh.xv[vert2] );
            dy     = abs(mesh.yv[vert1] - mesh.yv[vert2] );
            dAi    = sqrt(dx^2 + dy^2);

            # if iel==1
            #     println(bc)
            #     @printf("face node, x = %2.2e y = %2.2e\n", mesh.xf[nodei], mesh.yf[nodei])
            #     @printf("vert1    , x = %2.2e y = %2.2e\n", mesh.xv[vert1], mesh.yv[vert1])
            #     @printf("vert2    , x = %2.2e y = %2.2e\n", mesh.xv[vert2], mesh.yv[vert2])
            # end
           
            # Face normal
            n_x  =  dy/dAi
            n_y  = -dx/dAi
            
            # Third vector
            v_x  = mesh.xf[nodei] - mesh.xc[iel]
            v_y  = mesh.yf[nodei] - mesh.yc[iel]
            
            # Check wether the normal points outwards
            dot                 = n_x*v_x + n_y*v_y 
            mesh.n_x[iel,ifac]  = (dot>=0.0)*n_x - (dot<0.0)*n_x
            mesh.n_y[iel,ifac]  = (dot>=0.0)*n_y - (dot<0.0)*n_y
            mesh.dA[iel,ifac]   = dAi
        end
    end

    ############# FOR THE PSEUDO-TRANSIENT PURPOSES ONLY #############

    # Create face to element numbering
    mesh.f2e    = zeros(Int,mesh.nf,2)
    mesh.vole_f = zeros(Float64,mesh.nf,2)
    mesh.n_x_f  = zeros(Float64,mesh.nf,2)
    mesh.n_y_f  = zeros(Float64,mesh.nf,2)
    mesh.dA_f   = zeros(Float64,mesh.nf,2)

    # # Loop through field names and fields: standard
    # for fname in fieldnames(typeof(trimesh))
    #     println("Field name: ", fname)
    #     println("Content   : ", getfield(trimesh, fname))
    # end

    # Loop over edges and uses Voronoï diagram to get adjacent cells
    @tturbo for ifac=1:mesh.nf 
        mesh.f2e[ifac,1] = face2element[1,ifac]
        mesh.f2e[ifac,2] = face2element[2,ifac]
        act1 = face2element[1,ifac] > 0
        act2 = face2element[2,ifac] > 0
        iel1 = (act1==1) * face2element[1,ifac] + (act1==0) * 1
        iel2 = (act2==1) * face2element[2,ifac] + (act2==0) * 1
        # Compute face length
        vert1  = face2vertices[1,ifac]
        vert2  = face2vertices[2,ifac]
        xf     = 0.5*(mesh.xv[vert1] + mesh.xv[vert2])
        yf     = 0.5*(mesh.yv[vert1] + mesh.yv[vert2])
        dx     = (mesh.xv[vert1] - mesh.xv[vert2] );
        dy     = (mesh.yv[vert1] - mesh.yv[vert2] );
        dAi    = sqrt(dx^2 + dy^2);    
        mesh.dA_f[ifac,1] =  dAi
        mesh.dA_f[ifac,2] =  dAi
        # Volumes
        mesh.vole_f[ifac,1] = (act1==1) * mesh.vole[iel1]
        mesh.vole_f[ifac,2] = (act2==2) * mesh.vole[iel2]
        # Compute face normal
        n_x  = -dy/dAi
        n_y  =  dx/dAi
        # Third vector  (w.r.t. element 1)
        v_x  = xf - mesh.xc[iel1]
        v_y  = yf - mesh.yc[iel1]
        # Check wether the normal points outwards
        dot                 = n_x*v_x + n_y*v_y 
        mesh.n_x_f[ifac,1]  = (act1==1) * ((dot>=0.0)*n_x - (dot<0.0)*n_x)
        mesh.n_y_f[ifac,1]  = (act1==1) * ((dot>=0.0)*n_y - (dot<0.0)*n_y)
        # Take the negative for the second element
        mesh.n_x_f[ifac,2]  = -mesh.n_x_f[ifac,1]
        mesh.n_y_f[ifac,2]  = -mesh.n_y_f[ifac,1]
    end
    ############# FOR THE PSEUDO-TRANSIENT PURPOSES ONLY #############

    return mesh
end