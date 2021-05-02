using LoopVectorization

#--------------------------------------------------------------------#

function Generate2DRectMesh(nx,ny,xmin,xmax,ymin,ymax)
    # Generates a 2D rectangular mesh of nx*ny cells
    ncell      = nx*ny
    # Twice resolution mesh
    nx2 = 2*nx + 1
    ny2 = 2*ny + 1
    # 1D axis
    x2  = LinRange(xmin,xmax,nx2)
    y2  = LinRange(ymin,ymax,ny2)
    # 2D mesh: fake ngrid for P-T space ;)
    x2d   = repeat(x2, 1, length(y2))
    y2d   = repeat(y2, 1, length(x2))'
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
    # Face nodes
    xp = zeros(inum) 
    yp = zeros(inum)
    tp = zeros(Int64,inum) # node type - 1 boundary
    for i=1:nx2
        for j=1:ny2
            if (mod(j,2)==1 && mod(i,2)==0) || (mod(j,2)==0 && mod(i,2)==1)
                xp[nodes[i,j]] = x2d[i,j]
                yp[nodes[i,j]] = y2d[i,j]
                if (i==1 || i==nx2 || j==1 || j==ny2)
                    tp[nodes[i,j]] = 1
                end
            end
        end
    end
    # # South/North nodes
    # @avx for i=2:2:nx2-1
    #     for j=1:2:ny2
    #         xp[nodes[i,j]] = x2d[i,j]
    #         yp[nodes[i,j]] = y2d[i,j]
    #     end
    # end
    # # West/East nodes
    # @avx for i=1:2:nx2
    #     for j=2:2:ny2-1
    #         xp[nodes[i,j]] = x2d[i,j]
    #         yp[nodes[i,j]] = y2d[i,j]
    #     end
    # end
    # Cell 2 face numbering
    cell = zeros(4, ncell)
    @avx for i=1:nx
        for j=1:ny
            k  = i + (j-1)*nx
            ic = i+1 + (i-1)*1
            jc = j+1 + (j-1)*1
            cell[1,k] =  nodes[ic,jc-1] # South
            cell[2,k] =  nodes[ic+1,jc] # East
            cell[3,k] =  nodes[ic,jc+1] # North
            cell[4,k] =  nodes[ic-1,jc] # West
        end
    end
    return cell,xp,yp,tp
end

#--------------------------------------------------------------------#

# Create sides of mesh
xmin, xmax = 0, 1
ymin, ymax = 0, 1
nx, ny     = 5, 5  # number of elements

@time cell,xp,yp,tp = Generate2DRectMesh(nx,ny,xmin,xmax,ymin,ymax)
@time cell,xp,yp,tp = Generate2DRectMesh(nx,ny,xmin,xmax,ymin,ymax)
@time cell,xp,yp,tp = Generate2DRectMesh(nx,ny,xmin,xmax,ymin,ymax)
@time cell,xp,yp,tp = Generate2DRectMesh(nx,ny,xmin,xmax,ymin,ymax)