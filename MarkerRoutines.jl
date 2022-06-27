using Base.Threads

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

#--------------------------------------------------------------------#

function SetMarkers!( p, R )
    @tturbo for k=1:p.nmark
        in         = (p.x[k]^2 + p.y[k]^2) < R^2 
        p.phase[k] = (in==0)* 1.0 + (in==1)*2.0
    end
end

#--------------------------------------------------------------------#

function DoTheMarkerThing!( mesh, nx, ny, nmx, nmy, xmin, xmax, ymin, ymax, R, η, avg )
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
    LocateMarkers( p, dx ,dy, xc, yc, xmin, xmax, ymin, ymax)

    k2d = ones(ncx,ncy)
    @time Markers2Cells2( p, k2d, xc, yc, dx, dy, ncx, ncy, η, avg)
    if mesh.type=="Quadrangles" 
        mesh.ke .= k2d[:]
    end
    return nothing
end

#--------------------------------------------------------------------#

@views function Markers2Cells(p,phase,xc,yc,dx,dy,ncx,ncy)
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
                phase_th[tid][i, j] += p.phase[k] * area
                weight_th[tid][i, j] += area
                end
            end
        end
    end
    phase  .= reduce(+, phase_th)
    weight .= reduce(+, weight_th)
    phase ./= weight
    return
end

@views function CountMarkersPerCell(p,mpc)#,xc,yc,dx,dy,ncx,ncy)
mpc_th   = [similar(mpc) for _ = 1:nthreads()] # per thread
@threads for tid=1:nthreads()
    fill!(mpc_th[tid], 0)
end   
chunks = Iterators.partition(1:p.nmark, p.nmark ÷ nthreads())
    @sync for chunk in chunks
        Threads.@spawn begin
            tid = threadid()
            # fill!(mpc_th[tid], 0)
            for k in chunk
                # if p.phase[k]>=0
                i = p.cellx[k]
                j = p.celly[k]
                 # Get the column:
                # dstx = p.x[k] - xc[1];
                # i    = Int64(round(ceil( (dstx/dx) + 0.5)));
                # # Get the line:
                # dsty = p.y[k] - yc[1];
                # j    = Int64(round(ceil( (dsty/dy) + 0.5)));
                # Increment cell counts
                mpc_th[tid][i, j] += 1
                # end
            end
        end
    end
    fill!(mpc,0)
    mpc .= reduce(+, mpc_th)

    # # Count number of marker per cell
    # @avx for j=1:ncy, i=1:ncx
    #     mpc[i,j] = 0.0
    # end
    # @simd for k=1:p.nmark # @avx ne marche pas ici
    #     if (p.phase[k]>=0)
    #         # Get the column:
    #         dstx = p.x[k] - xc[1];
    #         i    = Int64(round(ceil( (dstx/dx) + 0.5)));
    #         # Get the line:
    #         dsty = p.y[k] - yc[1];
    #         j    = Int64(round(ceil( (dsty/dy) + 0.5)));
    #         # Increment cell count
    #         @inbounds mpc[i,j] += 1.0
    #     end
    # end
    return
end

#--------------------------------------------------------------------#

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

#--------------------------------------------------------------------#

@views function ListMarkers( p, ncx, ncy )
liste = hcat([[Int[] for i in 1:ncx] for j in 1:ncy]...)
for k in 1:p.nmark
    if p.phase[k]>=0
        i = p.cellx[k]
        j = p.celly[k]
        push!(liste[i,j], k)
    end
end
return liste
end

#--------------------------------------------------------------------#

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

#--------------------------------------------------------------------#

