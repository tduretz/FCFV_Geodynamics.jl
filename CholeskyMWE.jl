using LinearAlgebra, SparseArrays
import SuiteSparse

function Successive_solves!( x, Kc, b, nsolves )
    for i=1:nsolves
        x  .= Kc\b 
    end 
    return
end

function ThreePointStencil( N, dx )
    # Assemble matrix (not the part of interest)
    cC =  2.0/dx^2 * ones(Float64, N); cC[1]   = 3.0/dx^2; cC[end]   = 3.0/dx^2
    cW = -1.0/dx^2 * ones(Float64, N); cW[1]   = 0.0
    cE = -1.0/dx^2 * ones(Float64, N); cE[end] = 0.0
    iC = 1:N
    iW = ones(Int64, N); iW[2:end-0] .= iC[1:end-1]
    iE = ones(Int64, N); iE[1:end-1] .= iC[2:end-0]
    I  = [iC[:]; iC[:]; iC[:]]
    J  = [iC[:]; iW[:]; iE[:]]
    V  = [cC[:]; cW[:]; cE[:]]
    K  = sparse( I, J, V )
    droptol!(K, 1e-9)
    return K
end


# Input
N  = 10000000
dx = 1

# Sparse matrix
K = ThreePointStencil( N, dx )

# Allocate arrays
b = rand(N)           # Right-hand-side array 
x = zeros(Float64, N) # Solution array

# Factor
Kc = cholesky(K)

for i=1:3
@time @views x  .= Kc\b
@time @views x  .= SuiteSparse.CHOLMOD.solve(0, Kc, SuiteSparse.CHOLMOD.Dense(b) )[:]
end

# SuiteSparse.CHOLMOD.spsolve(Kc,SparseArrays.CHOLMOD.Sparse(b))
# ldiv!(Kc, b)

# # One solve
# @time x  .= Kc\b  # why does this allocate memory although the factorisation done and the vectors (x, b) are allocated?

# # Successive solves
# nsolves = 10
# @time Successive_solves!( x, Kc, b, nsolves ) # then, memory allocation depends on the number of solves...