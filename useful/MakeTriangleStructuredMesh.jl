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
end


 # Create sides of mesh
     xmin, xmax = 0, 1
     ymin, ymax = 0, 1
     nx, ny     = 50, 50
 
 # Generates a 2D rectangular mesh of nx*ny cells * 2 triangles
 ncell      = nx*ny*2
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
         if (mod(j,2)==1 && mod(i,2)==0) || (mod(j,2)==0 && mod(i,2)==1) ||  (mod(j,2)==0 && mod(i,2)==0)
             global inum += 1
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
            global inum += 1
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
         if (mod(j,2)==1 && mod(i,2)==0) || (mod(j,2)==0 && mod(i,2)==1) ||  (mod(j,2)==0 && mod(i,2)==0)
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
 # Cell 2 face numbering - for matrix connectivity - lower triangles
 face = zeros(Int64, 3, ncell)
 @avx for i=1:nx
     for j=1:ny
         k  = j + (i-1)*ny
         jc = i+1 + (i-1)*1
         ic = j+1 + (j-1)*1
         face[1,k] =  nodes[ic,jc-1] # South
         face[2,k] =  nodes[ic-1,jc] # West
         face[3,k] =  nodes[ic  ,jc] # centerh
         # println(face[:,k])
     end
 end
  # Cell 2 face numbering - for matrix connectivity - upper triangles
 @avx for i=1:nx
    for j=1:ny
        k  = j + (i-1)*ny + nx*ny
        jc = i+1 + (i-1)*1
        ic = j+1 + (j-1)*1
        face[1,k] =  nodes[ic+1,jc] # East
        face[2,k] =  nodes[ic,jc+1] # North
        face[3,k] =  nodes[ic  ,jc] # center
        # println(face[:,k])
    end
end
 # Cell 2 vertices - used for visualisation of quads - lower triangles
 vert = zeros(Int64, 3, ncell)
 @avx for i=1:nx
     for j=1:ny
         k  = j + (i-1)*ny
         jc = i+1 + (i-1)*1
         ic = j+1 + (j-1)*1
         vert[1,k] =  nodes[ic-1,jc+1] 
         vert[2,k] =  nodes[ic-1,jc-1] 
         vert[3,k] =  nodes[ic+1,jc-1] 
        #  vert[4,k] =  nodes[ic+1,jc+1] 
     end
 end
  # Cell 2 vertices - used for visualisation of quads - lower triangles
  @avx for i=1:nx
      for j=1:ny
          k  = j + (i-1)*ny + nx*ny
          jc = i+1 + (i-1)*1
          ic = j+1 + (j-1)*1
          vert[1,k] =  nodes[ic-1,jc+1] 
          vert[2,k] =  nodes[ic+1,jc+1] 
          vert[3,k] =  nodes[ic+1,jc-1]
      end
  end
 # Centroids
 xc = zeros(ncell)
 yc = zeros(ncell)
 w  = 1.0/3.0
 @avx for iel=1:ncell
     tempx = 0
     tempy = 0
     for j=1:3
         tempx += w * xf[face[j,iel]]
         tempy += w * yf[face[j,iel]]
     end
     xc[iel] = tempx
     yc[iel] = tempy
     # println(xc[iel])
 end

 # Fill structure
 mesh        = FCFV_Mesh()
 mesh.type   = "triangle_struct"
 mesh.nel    = ncell
 mesh.nf     = nface
 mesh.nf_el  = 3
 mesh.nf_el  = 3
 mesh.nv     = (nx+1)*(ny+1)
 mesh.xv     = xn
 mesh.yv     = yn
 mesh.xf     = xf
 mesh.yf     = yf
 mesh.e2v    = vert' # cell nodes - not dofs! Dofs are in 'cell'
 mesh.e2f    = face'
 mesh.xc     = xc
 mesh.yc     = yc
 mesh.vole   = dx*dy*ones(ncell)/2
 mesh.bc     = tf

 nodeA = [2 1 3]
 nodeB = [3 2 1]

 # Compute normal to faces
 mesh.n_x = zeros(Float64,mesh.nel,mesh.nf_el)
 mesh.n_y = zeros(Float64,mesh.nel,mesh.nf_el)
 mesh.dA  = zeros(Float64,mesh.nel,mesh.nf_el)

  # Assemble FCFV elements
  for iel=1:mesh.nel  
     
     # println("element: ",  iel)

     for ifac=1:mesh.nf_el
         
         nodei  = mesh.e2f[iel,ifac]

         # Vertices
         vert1  = mesh.e2v[iel,nodeA[ifac]]
         vert2  = mesh.e2v[iel,nodeB[ifac]]
         bc     = mesh.bc[nodei]
         dxl     = abs(mesh.xv[vert1] - mesh.xv[vert2] );
         dyl     = abs(mesh.yv[vert1] - mesh.yv[vert2] );
         dAi    = sqrt(dxl^2 + dyl^2);

        #  if iel==1 || iel==(nx*ny+1)
        #      println(bc)
        #      @printf("face node, x = %2.2e y = %2.2e\n", mesh.xf[nodei], mesh.yf[nodei])
        #      @printf("vert1    , x = %2.2e y = %2.2e\n", mesh.xv[vert1], mesh.yv[vert1])
        #      @printf("vert2    , x = %2.2e y = %2.2e\n", mesh.xv[vert2], mesh.yv[vert2])
        #  end
        
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