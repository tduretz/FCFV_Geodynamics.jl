import Statistics
using AlgebraicMultigrid
import IterativeSolvers
using SuiteSparse

function StokesSolvers(mesh, Kuu, Kup, fu, fp, M, solver)
    if solver==0
        # Coupled solve
        zero_p = spdiagm(mesh.nel, mesh.nel) 
        K      = [Kuu Kup; -Kup' zero_p]
        f      = [fu; fp]
        # xh     = lu(K)\f
        xh      = K\f
        Vxh    = xh[1:mesh.nf]
        Vyh    = xh[mesh.nf+1:2*mesh.nf]
        Pe     = xh[2*mesh.nf+1:end] 
        Pe     = Pe .- Statistics.mean(Pe)
    elseif solver==1
        # Decoupled solve
        coef  = 1e4.*mesh.ke./mesh.Ω#*ones(mesh.nel)
        # file = matopen(string(@__DIR__,"/results/matrix_ppi.mat"), "w" )
        # write(file, "coef",    coef )
        # close(file)
        Kppi  = spdiagm(coef)
        Kpu   = -Kup'
        Kuusc = Kuu .- Kup*(Kppi*Kpu)
        # PC    = Kuu .- diag(Kuu) .+ diag(M)

        # PC    =  0.5*(M .+ M') 
        # Kuusc1 = PC .- Kup*(Kppi*Kpu)
        # PC     =  0.5*(Kuusc1 .+ Kuusc1') 
        # t = @elapsed Kf    = cholesky(Hermitian(PC),check = false)
        t = @elapsed Kf    = lu(Kuusc)
        # @printf("Cholesky took = %02.2e s\n", t)
        u     = zeros(2*mesh.nf,1)
        ru    = zeros(2*mesh.nf, 1)
        fusc  = zeros(2*mesh.nf,1)
        p     = zeros(mesh.nel, 1)
        rp    = zeros(mesh.nel, 1)
        # Iterations
        for rit=1:20
            ru   .= fu .- Kuu*u .- Kup*p;
            rp   .= fp .- Kpu*u;
            nrmu = norm(ru)
            nrmp = norm(rp)
            @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
            if nrmu/sqrt(length(ru)) < 1e-13 && nrmp/sqrt(length(ru)) < 1e-13
                break
            end
            fusc .= fu  .- Kup*(Kppi*fp .+ p)
            u    .= Kf\fusc
            # u    .= Kuusc\fusc
            # KSP_GCR_StokesFCFV!( u, Kuusc, fusc, 1e-10, 2, Kxxf, f, v, s, val, VV, SS, restart  )
            p   .+= Kppi*(fp .- Kpu*u)
        end
        # Post-process solve
        Vxh = u[1:mesh.nf]
        Vyh = u[mesh.nf+1:2*mesh.nf]
        Pe  = p[:]
    elseif solver==2
        # Decoupled solve
        coef  = 1e3.*mesh.ke./mesh.vole#*ones(mesh.nel)
        # file = matopen(string(@__DIR__,"/results/matrix_ppi.mat"), "w" )
        # write(file, "coef",    coef )
        # close(file)
        Kppi  = spdiagm(coef)
        Kpu   = -Kup'
        Kuusc = Kuu - Kup*(Kppi*Kpu)
        PC    =  0.5*(Kuusc + Kuusc') 
        # t = @elapsed Kf    = cholesky(Hermitian(PC),check = false)
        # @time ml = ruge_stuben(PC) #pas mal
        @time ml = smoothed_aggregation(PC) #pas mal
        @time pc = aspreconditioner(ml)
        # @printf("Cholesky took = %02.2e s\n", t)
        u     = zeros(2*mesh.nf, 1)
        ru    = zeros(2*mesh.nf, 1)
        fusc  = zeros(2*mesh.nf, 1)
        p     = zeros( mesh.nel,  1)
        rp    = zeros( mesh.nel,  1)
        # Iterations
        for rit=1:20
            ru   .= fu .- Kuu*u .- Kup*p
            rp   .= fp .- Kpu*u
            @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, norm(ru)/sqrt(length(ru)), norm(rp)/sqrt(length(rp)))
            if norm(ru)/sqrt(length(ru)) < 1e-12 && norm(rp)/sqrt(length(ru)) < 1e-12
                break
            end
            fusc .=  fu  - Kup*(Kppi*fp + p)
            # u    .= Kf\fusc
            # @time u = IterativeSolvers.gmres(Kuusc, fusc, Pl = pc, reltol=1e-6) #, maxiter=100
            # @time IterativeSolvers.cg!(u, Kuusc, fusc, Pl = pc, reltol=1e-3)
            @time IterativeSolvers.cg!(u, PC, fusc, Pl = pc, reltol=1e-6)
            p   .+= Kppi*(fp - Kpu*u)
        end
        # Post-process solve
        Vxh = u[1:mesh.nf]
        Vyh = u[mesh.nf+1:2*mesh.nf]
        Pe  = p[:]
    elseif solver==3
        # Decoupled solve
        coef  = 1e3.*mesh.ke./mesh.Ω#*ones(mesh.nel)
        Kppi  = spdiagm(coef)
        Kpu   = .- Kup'
        Kuusc = Kuu .- Kup*(Kppi*Kpu)
        ndof  = size(Kuu,1)
        ndofx = Int64(ndof/2)
        Kxx   = M[1:ndofx,1:ndofx]
        t = @elapsed Kxxf  = cholesky(Hermitian(Kxx),check = false)
        @printf("Cholesky took = %02.2e s\n", t)
        u     = zeros(Float64, 2*mesh.nf)
        ru    = zeros(Float64, 2*mesh.nf)
        fusc  = zeros(Float64, 2*mesh.nf)
        p     = zeros(Float64, mesh.nel)
        rp    = zeros(Float64, mesh.nel)
        ######################################
        restart = 30
        f      = zeros(Float64, 2*mesh.nf)
        v      = zeros(Float64, 2*mesh.nf)
        s      = zeros(Float64, 2*mesh.nf)
        val    = zeros(Float64, restart)
        VV     = zeros(Float64, (2*mesh.nf, restart) )  # !!!!!!!!!! allocate in the right sense :D
        SS     = zeros(Float64, (2*mesh.nf, restart) )
        ######################################
        # Iterations
        for rit=1:20
            ru   .= fu[:] .- Kuu*u .- Kup*p
            rp   .= fp[:] .- Kpu*u
            nrmu = sqrt(mydotavx( ru,ru ) )#norm(v)
            nrmp = sqrt(mydotavx( rp,rp ) )#norm(v)
            @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
            if nrmu/sqrt(length(ru)) < 1e-10 && nrmp/sqrt(length(ru)) < 1e-10
                break
            end
            fusc .=  fu[:] .- Kup*(Kppi*fp .+ p)
            @time KSP_GCR_StokesFCFV!( u, Kuusc, fusc, 1e-10, 2, Kxxf, f, v, s, val, VV, SS, restart  )
            p   .+= Kppi*(fp .- Kpu*u)
        end
        # Post-process solve
        Vxh = u[1:mesh.nf]
        Vyh = u[mesh.nf+1:2*mesh.nf]
        Pe  = p[:]
    end
    return Vxh, Vyh, Pe
end

#--------------------------------------------------------------------#

function KSP_GCR_StokesFCFV!( x::Vector{Float64}, M::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, eps::Float64, noisy::Int64, Kxxf::SuiteSparse.CHOLMOD.Factor{Float64}, f::Vector{Float64}, v::Vector{Float64}, s::Vector{Float64}, val::Vector{Float64}, VV::Matrix{Float64}, SS::Matrix{Float64}, restart::Int64 )
    # Initialise
    val .= 0.0
    s   .= 0.0
    v   .= 0.0
    VV  .= 0.0
    SS  .= 0.0
    # KSP GCR solver
    norm_r, norm0 = 0.0, 0.0
    N               = length(x)
    maxit           = 10*restart
    ncyc, its       = 0, 0
    i1, i2, success = 0, 0, 0
    # Initial residual
    @tturbo f     .= b .- M*x 
    norm_r = sqrt(mydotavx( f, f ) )#norm(v)norm(f)
    norm0  = norm_r;
    ndof   = size(M,1)
    ndofu  = Int64(ndof/2)
    # ldiv!(P::SuiteSparse.CHOLMOD.Factor{Float64}, v) = (P \ v)
    # Solving procedure
     while ( success == 0 && its<maxit ) 
        for i1=1:restart
            # Apply preconditioner, s = PC^{-1} f
            s[1:ndofu]     .= Kxxf \ @view f[1:ndofu]
            s[ndofu+1:end] .= Kxxf \ @view f[ndofu+1:end]
            # Action of Jacobian on s: v = J*s
            @tturbo mul!(v, M, s)
            # Approximation of the Jv product
            for i2=1:i1
                val[i2] = mydotavx( v, view(VV, :, i2 ) )   
            end
            # Scaling
            for i2=1:i1
                @tturbo v .-= val[i2] .* view(VV, :, i2 )
                @tturbo s .-= val[i2] .* view(SS, :, i2 )
            end
            # -----------------
            nrm_inv = 1.0 / sqrt(mydotavx( v, v ) )
            r_dot_v = mydotavx( f, v )  * nrm_inv
            # -----------------
            @tturbo v     .*= nrm_inv
            @tturbo s     .*= nrm_inv
            # -----------------
            @tturbo x     .+= r_dot_v.*s
            @tturbo f     .-= r_dot_v.*v
            # -----------------
            norm_r  = sqrt(mydotavx( f, f ) )
            if norm_r/sqrt(length(f)) < eps || its==23
                @printf("It. %04d: res. = %2.2e\n", its, norm_r/sqrt(length(f)))
                success = 1
                println("converged")
                break
            end
            # Store 
            @tturbo VV[:,i1] .= v
            @tturbo SS[:,i1] .= s
            its      += 1
        end
        its  += 1
        ncyc += 1
    end
    if (noisy>1) @printf("[%1.4d] %1.4d KSP GCR Residual %1.12e %1.12e\n", ncyc, its, norm_r, norm_r/norm0); end
    return its
end
export KSP_GCR_Stokes!

#--------------------------------------------------------------------#

function mydotavx(A, B)
    s = zero(promote_type(eltype(A), eltype(B)))
    @tturbo for i in eachindex(A,B)
        s += A[i] * B[i]
    end
    s
end