import Statistics:mean
using AlgebraicMultigrid
import IterativeSolvers
using SuiteSparse

function StokesSolvers!(Vxh, Vyh, Pe, mesh, Kuu, Kup, Kpu, Kpp, fu, fp, M, solver; penalty=1e4, tol=1e-10, comp=false)

    @printf("Start solver %d\n", solver)
    nVx = Int64(length(fu)/2)
    nVy = length(fu)
    nV  = length(fu)
    nP  = length(fp)
    @printf("ndof = %d\n", nV+nP)
    
    if solver==0
        # Coupled solve
        K      = [Kuu Kup; Kpu Kpp]
        f      = [fu; fp]
        xh     = K\f
        Vxh   .= xh[1:nVx]
        Vyh   .= xh[nVx+1:nVy]
        Pe    .= xh[nVy+1:end] 
        if comp==false Pe    .= Pe .- mean(Pe) end
    elseif solver==1 || solver==-1
        # Decoupled solve
        if comp==false 
            coef  = zeros(mesh.nel*mesh.npel)
            for i=1:mesh.npel
                coef[(i-1)*mesh.nel+1:i*mesh.nel] .= penalty.*mesh.ke./mesh.Ω
            end
            Kppi  = spdiagm(coef)
        else
            Kppi  = spdiagm( 1.0 ./ diag(Kpp) )
        end
        Kuusc = Kuu .- Kup*(Kppi*Kpu)
        # PC    = Kuu .- diag(Kuu) .+ diag(M)
        # PC    =  0.5*(M .+ M') 
        # Kuusc1 = PC .- Kup*(Kppi*Kpu)
        # PC     =  0.5*(Kuusc1 .+ Kuusc1') 
        if solver==-1 
            t = @elapsed Kf    = cholesky(Hermitian(Kuusc),check = false)
            @printf("Cholesky took = %02.2e s\n", t)
        else
            t = @elapsed Kf    = lu(Kuusc)
            @printf("LU took = %02.2e s\n", t)
        end
        u     = zeros(length(fu), 1)
        ru    = zeros(length(fu), 1)
        fusc  = zeros(length(fu), 1)
        p     = zeros(length(fp), 1)
        rp    = zeros(length(fp), 1)
        # Iterations
        for rit=1:20
            ru   .= fu .- Kuu*u .- Kup*p
            rp   .= fp .- Kpu*u .- Kpp*p
            nrmu = norm(ru)
            nrmp = norm(rp)
            @printf("  --> Powell-Hestenes Iteration %02d\n  Momentum res.   = %2.2e\n  Continuity res. = %2.2e\n", rit, nrmu/sqrt(length(ru)), nrmp/sqrt(length(rp)))
            if nrmu/sqrt(length(ru)) < tol && nrmp/sqrt(length(ru)) < tol
                break
            end
            fusc .= fu  .- Kup*(Kppi*fp .+ p)
            u    .= Kf\fusc
            # @time KSP_GCR_StokesFCFV!( u, Kuusc, fusc, tol, 2, Kxxf, f, v, s, val, VV, SS, restart  )
            p   .+= Kppi*(fp .- Kpu*u .- Kpp*p)
        end
        # Post-process solve
        Vxh .= u[1:nVx]
        Vyh .= u[nVx+1:nVy]
        Pe  .= p[:]
    end
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
     f     .= b .- M*x 
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
             mul!(v, M, s)
            # Approximation of the Jv product
            for i2=1:i1
                val[i2] = mydotavx( v, view(VV, :, i2 ) )   
            end
            # Scaling
            for i2=1:i1
                 v .-= val[i2] .* view(VV, :, i2 )
                 s .-= val[i2] .* view(SS, :, i2 )
            end
            # -----------------
            nrm_inv = 1.0 / sqrt(mydotavx( v, v ) )
            r_dot_v = mydotavx( f, v )  * nrm_inv
            # -----------------
             v     .*= nrm_inv
             s     .*= nrm_inv
            # -----------------
             x     .+= r_dot_v.*s
             f     .-= r_dot_v.*v
            # -----------------
            norm_r  = sqrt(mydotavx( f, f ) )
            if norm_r/sqrt(length(f)) < eps || its==23
                @printf("It. %04d: res. = %2.2e\n", its, norm_r/sqrt(length(f)))
                success = 1
                println("converged")
                break
            end
            # Store 
             VV[:,i1] .= v
             SS[:,i1] .= s
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
     for i in eachindex(A,B)
        s += A[i] * B[i]
    end
    s
end