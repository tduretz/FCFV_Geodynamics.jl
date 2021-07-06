using Base.Threads

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
