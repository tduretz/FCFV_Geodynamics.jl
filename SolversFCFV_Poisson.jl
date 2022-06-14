using AlgebraicMultigrid
import IterativeSolvers: cg
# import Metis

#--------------------------------------------------------------------#

function SolvePoisson(mesh, K, f, solver)
    # Solve for hybrid variable
    if solver == 0
        @time Th   = K\f
    elseif solver == 1
        println("Direct solve:")
        PC  = 0.5.*(K.+K')
        # @time permetis, iperm = Metis.permutation(PC)
        # PCc = cholesky(PC, perm=convert(Vector{Int64},permetis))
        PCc = cholesky(PC)
        Th  = zeros(mesh.nf)
        dTh = zeros(mesh.nf,1)
        r   = zeros(mesh.nf,1)
        r  .= f - K*Th
        # @time Th   = K\f
        for rit=1:5
            r    .= f - K*Th
            println("It. ", rit, " - Norm of residual: ", norm(r)/length(r))
            if norm(r)/length(r) < 1e-10
                break
            end
            dTh  .= PCc\r
            Th  .+= dTh[:]
        end
    elseif solver == 2
        println("AMG preconditionned CG solver:")
        # ml = ruge_stuben(K)
        @time ml = smoothed_aggregation(K) #pas mal
        @time p  = aspreconditioner(ml)
        @time Th = cg(K, f, Pl = p, reltol=1e-6)
    end
    return Th
    end