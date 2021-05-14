mutable struct Mesh
    type  :: Symbol
    n     ::Ref{Int64}
    x     ::Vector{Float64}
    Mesh() = new( )
end

# Create structure with default missing fields
mesh = Mesh()

mesh.n  = 101
mesh.x  = [1.1, 1.5, 1.9]

println( mesh.n )
println( mesh.x )