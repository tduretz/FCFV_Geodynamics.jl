using CUDA
using CellArrays, StaticArrays

@inbounds function init_CellArrays!(K,Q,K_glob,Q_glob,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        for c2 = 1:cellsize(K,2), c1 = 1:cellsize(K,1)
            field(K,c1,c2)[ie] = K_glob[ie,c1,c2]
            field(Q,c1   )[ie] = Q_glob[ie,c1]
        end
    end
    return nothing
end

@inbounds function my_gpu_fun_0!(V,P,Vx,Vy,Pr,e2n,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        nodes = e2n[ie,:]
        field(V,1)[ie] = Vx[nodes[1]]
        field(V,3)[ie] = Vx[nodes[2]]
        field(V,2)[ie] = Vy[nodes[1]]
        field(V,4)[ie] = Vy[nodes[2]]
        field(P,1)[ie] = Pr[ie]
    end
    return nothing
end

@inbounds function my_gpu_fun_1!(Fv,Fp,K,Q,V,P,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        Fv[ie] = K[ie]*V[ie] .+ Q[ie]  * P[ie]
        Fp[ie] =             .- Q[ie]' * V[ie]
    end
    return nothing
end

@inbounds function my_gpu_fun_2!(Fx,Fy,FP,Fv,Fp,e2n,nel)
    ie = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ie <= nel
        nodes = e2n[ie,:]
        CUDA.@atomic Fx[nodes[1]] -= Fv[ie][1]
        CUDA.@atomic Fx[nodes[2]] -= Fv[ie][3]
        CUDA.@atomic Fy[nodes[1]] -= Fv[ie][2]
        CUDA.@atomic Fy[nodes[2]] -= Fv[ie][4]
                     FP[ie]       -= Fp[ie][1]
    end
    return nothing
end

function test_kernels()
    DAT      = Float64
    blocks   = 1
    threads  = 10
    nel      = blocks*threads
    nn       = nel+1
    nnel     = 2
    npel     = 1
    dims_nel = (nel)
    # local cell array size
    celldims_K_loc   = (nnel*2, nnel*2)
    celldims_Q_loc   = (nnel*2, 1)
    celldims_V_loc   = (nnel*2, 1)
    celldims_P_loc   = (npel  , 1)
    # init FEM tmp CellArrays
    K_loc  = SMatrix{celldims_K_loc..., DAT, prod(celldims_K_loc)}
    K      = CuCellArray{K_loc}(undef, dims_nel)
    Q_loc  = SMatrix{celldims_Q_loc..., DAT, prod(celldims_Q_loc)}
    Q      = CuCellArray{Q_loc}(undef, dims_nel)
    V_loc  = SMatrix{celldims_V_loc..., DAT, prod(celldims_V_loc)}
    V      = CuCellArray{V_loc}(undef, dims_nel)
    P_loc  = SMatrix{celldims_P_loc..., DAT, prod(celldims_P_loc)}
    P      = CuCellArray{P_loc}(undef, dims_nel)
    Fv_loc = SMatrix{celldims_V_loc..., DAT, prod(celldims_V_loc)}
    Fv     = CuCellArray{Fv_loc}(undef, dims_nel)
    Fp_loc = SMatrix{celldims_P_loc..., DAT, prod(celldims_P_loc)}
    Fp     = CuCellArray{Fp_loc}(undef, dims_nel)
    # element to node numbering
    e2n = SMatrix{nel, 2, Int64}([1 2 3 4 5 6 7 8 9 10;
                                  2 3 4 5 6 7 8 9 10 11]')
    # init data
    K_glob = 4.0 .* CUDA.ones(Float64,nel,nnel*2,nnel*2)
    Q_glob = 2.0 .* CUDA.ones(Float64,nel,nnel*2,1     )
    @cuda blocks=blocks threads=threads init_CellArrays!(K,Q,K_glob,Q_glob,nel); synchronize()

    # K[1] = @SMatrix rand(nnel*2, nnel*2)

    Fx = CUDA.zeros(DAT, nn)
    Fy = CUDA.zeros(DAT, nn)
    FP = CUDA.zeros(DAT, nel*npel)
    Vx = CUDA.zeros(DAT, nn)
    Vy = CUDA.zeros(DAT, nn)
    Pr = CUDA.zeros(DAT, nel*npel)

    @time begin
        @cuda blocks=blocks threads=threads my_gpu_fun_0!(V,P,Vx,Vy,Pr,e2n,nel); synchronize()
        @cuda blocks=blocks threads=threads my_gpu_fun_1!(Fv,Fp,K,Q,V,P,nel); synchronize()
        @cuda blocks=blocks threads=threads my_gpu_fun_2!(Fx,Fy,FP,Fv,Fp,e2n,nel); synchronize()
    end
    return
end

test_kernels()
