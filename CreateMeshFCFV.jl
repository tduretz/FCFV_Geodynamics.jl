import TriangleMesh
using Printf
using LoopVectorization

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
    f2e    ::Union{Matrix{Int64},   Missing} = missing # face 2 element numbering
    dA_f   ::Union{Matrix{Float64}, Missing} = missing # face 2 element numbering
    vole_f ::Union{Matrix{Float64}, Missing} = missing # volume of element
    n_x_f  ::Union{Matrix{Float64}, Missing} = missing # normal 2 face x
    n_y_f  ::Union{Matrix{Float64}, Missing} = missing # normal 2 face y
end

function MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax )

dx   = (xmax-xmin)/nx
dy   = (ymax-ymin)/ny
area = dx*dy

# Four corners of the domain
px   = [xmin xmax xmax xmin]
py   = [ymin ymin ymax ymax]
sx   = [ 1 2 3 4 ] 
sy   = [ 2 3 4 1 ]
st   = [ 1 1 1 1 ]          # segment markers

p    = vcat(px, py)         # points
s    = vcat(sx, sy)         # segments

# inum = 0;
# dx
# px = Float64[]
# py = Float64[]
# sx = Int64[]
# sy = Int64[]
# st = Int64[]
# for i=1:nx+1
#     inum+=1
#     px   = push!(px, (i-1)*dx)
#     py   = push!(py,    ymin)
#     sx   = push!(sx, inum   )
#     sy   = push!(sy, inum+1 )
#     st   = push!(st, 1      )
# end

# for i=1:nx+1
#     inum+=1
#     px   = push!(px, (i-1)*dx)
#     py   = push!(py,    ymax)
#     sx   = push!(sx, inum   )
#     sy   = push!(sy, inum+1 )
#     st   = push!(st, 1      )
# end

# for i=1:ny+1
#     inum+=1
#     py   = push!(py, (i-1)*dy)
#     px   = push!(px,    xmin)
#     sx   = push!(sx, inum   )
#     sy   = push!(sy, inum+1 )
#     st   = push!(st, 1      )
# end

# for i=1:ny+1
#     inum+=1
#     py   = push!(py, (i-1)*dy)
#     px   = push!(px,    xmax)
#     sx   = push!(sx, inum   )
#     sy   = push!(sy, inum+1 )
#     st   = push!(st, 1      )
# end

# sy[end] = 1;

# p    = hcat(px, py)         # points
# s    = hcat(sx, sy)         # segments

# p = p'
# s = s'

st = st[:]

# Triangulation
holes    = Array{Float64}(undef,2,0)
domain   = TriangleMesh.Polygon_pslg(size(p,2), p, 0, Array{Int64}(undef,2,0), 0, Array{Float64}(undef,2,0),  size(s,2), s, st, 0, holes)
astring = @sprintf("%0.10lf", area)
switches = "QDpenq33o2IAa$(astring)"

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

nel  = trimesh.n_cell
vole = zeros(nel)
xc   = zeros(nel)
yc   = zeros(nel)

@avx for iel=1:nel
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

 # Assemble FCFV elements
 @avx for iel=1:mesh.nel  
    
    # println("element: ",  iel)

    for ifac=1:mesh.nf_el
        
        nodei  = mesh.e2f[iel,ifac]

        # Vertices
        vert1  = mesh.e2v[iel,nodeA[ifac]]
        vert2  = mesh.e2v[iel,nodeB[ifac]]
        vert3  = mesh.e2v[iel,nodeC[ifac]]
        bc     = mesh.bc[nodei]
        dx     = (mesh.xv[vert1] - mesh.xv[vert2] );
        dy     = (mesh.yv[vert1] - mesh.yv[vert2] );
        dAi    = sqrt(dx^2 + dy^2);

        # println(bc)
        # @printf("face node, x = %2.2e y = %2.2e\n", mesh.xf[nodei], mesh.yf[nodei])
        # @printf("vert1    , x = %2.2e y = %2.2e\n", mesh.xv[vert1], mesh.yv[vert1])
        # @printf("vert2    , x = %2.2e y = %2.2e\n", mesh.xv[vert2], mesh.yv[vert2])
       
        # Face normal
        n_x  = -dy/dAi
        n_y  =  dx/dAi
        
        # Third vector
        v_x  = mesh.xf[nodei] - mesh.xc[iel]
        v_y  = mesh.yf[nodei] - mesh.yc[iel]
        # v_x  = mesh.xv[vert1] - mesh.xv[vert3]
        # v_y  = mesh.yv[vert1] - mesh.yv[vert3]
        
        # Check wether the normal points outwards
        dot                 = n_x*v_x + n_y*v_y 
        mesh.n_x[iel,ifac]  = ((dot>=0.0)*n_x - (dot<0.0)*n_x)
        mesh.n_y[iel,ifac]  = ((dot>=0.0)*n_y - (dot<0.0)*n_y)
        mesh.dA[iel,ifac]   = dAi
    end
end

return mesh

end

#--------------------------------------------------------------------#

function MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax)
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
                if (i==1 || i==nx2 || j==1 || j==ny2)
                    tf[nodes[i,j]] = 1
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
    @avx for i=1:nx
        for j=1:ny
            k  = j + (i-1)*ny
            jc = i+1 + (i-1)*1
            ic = j+1 + (j-1)*1
            face[1,k] =  nodes[ic,jc-1] # South
            face[2,k] =  nodes[ic-1,jc] # West
            face[3,k] =  nodes[ic+1,jc] # East
            face[4,k] =  nodes[ic,jc+1] # North
            # println(face[:,k])
        end
    end
    # Cell 2 vertices - used for visualisation of quads
    vert = zeros(Int64, 4, ncell)
    @avx for i=1:nx
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
    @avx for iel=1:ncell
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
    mesh        = FCFV_Mesh()
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

    nodeA = [2 1 3 4]
    nodeB = [3 2 4 1]

    # Compute normal to faces
    mesh.n_x = zeros(Float64,mesh.nel,mesh.nf_el)
    mesh.n_y = zeros(Float64,mesh.nel,mesh.nf_el)
    mesh.dA  = zeros(Float64,mesh.nel,mesh.nf_el)

     # Assemble FCFV elements
    @avx for iel=1:mesh.nel  
        
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


    return mesh
end