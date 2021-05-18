Base.@kwdef mutable struct Mesh
    type   ::Union{String, Missing}          = missing
    nc     ::Union{Int64,  Missing}          = missing
    nn     ::Union{Int64,  Missing}          = missing
    nn_el  ::Union{Int64,  Missing}          = missing
    nf_el  ::Union{Int64,  Missing}          = missing
    xn     ::Union{Vector{Float64}, Missing} = missing # node x coordinate
    yn     ::Union{Vector{Float64}, Missing} = missing # node y coordinate
    xf     ::Union{Vector{Float64}, Missing} = missing # face x coordinate
    yf     ::Union{Vector{Float64}, Missing} = missing # face y coordinate
    c2n    ::Union{Vector{Int64},   Missing} = missing # cell 2 node numbering
    c2f    ::Union{Vector{Int64},   Missing} = missing # cell 2 face numbering
    xyviz  ::Union{Vector{Float64}, Missing} = missing # table of coordinates for triangle visualisation with PyPlot
    numviz ::Union{Vector{Float64}, Missing} = missing # table of numbering for triangle visualisation with PyPlot
end

# Create structure with default missing fields
mesh = Mesh() 

# Check if value is missing
ismissing(mesh.xn)

# Assign value to fields
mesh.type = "triangle"
mesh.nc   = 101
mesh.xf   = [1.1, 1.5, 1.9]

# Assigns value based on attribute value
if mesh.type == "triangle"
    mesh.nn_el = 3
    mesh.nf_el = 3
end
  
# Loop through field names and fields: standard
for fname in fieldnames(typeof(mesh))
    println("Field name: ", fname)
    println("Content   : ", getfield(mesh, fname))
end

# Loop through struct fields: meta-programming  
@generated function test4(p::P) where P
    assignments = [
        :( println("Field name: ", p.$name, "\n Content: ")  ) for name in fieldnames(P) #println(p.$name)
    ]
    quote $(assignments...) end
end
test4(mesh)