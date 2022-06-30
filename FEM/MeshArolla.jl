using Printf, DelimitedFiles, Plots

include("../CreateMeshFCFV.jl")
include("../VisuFCFV.jl")
#---------------------------------#
function StabParam(τ, Γ, Ω, mesh_type, ν) 
    return 0. # Stabilisation is only needed for FCFV
end
#---------------------------------#

data = readdlm("./FEM/arolla51.txt", ' ')

xp = zeros(2*size(data,1)) 
yp = zeros(2*size(data,1))
xp[1:size(data,1)]     .= data[:,1]
xp[size(data,1)+1:end] .= reverse(data[:,1])
yp[1:size(data,1)]     .= data[:,2]
yp[size(data,1)+1:end] .= reverse(data[:,3])

# somehow need to subsample the data
xp = xp[2:2:end]
yp = yp[2:2:end]

xmin, xmax = minimum(xp), maximum(xp)
ymin, ymax = minimum(yp), maximum(yp)
# xp = [xmin+eps(); xmax-eps(); xmax-eps(); xmin+eps() ]
# yp = [ymin+eps(); ymin+eps(); ymax-eps(); ymax-eps() ]

p = Plots.scatter( xp./1e3, yp./1e3, label="Input")
display(Plots.plot(p, ylims=(2,3.5), aspect_ratio=1.0, xlabel="x [km]", ylabel="y [km]"))

nx, ny     = 2, 2
model      = -1
area       = 2000.0
mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax, 0.0, model, 0.0, 0, area; xp_in=xp, yp_in=yp ) 
println("Number of elements: ", mesh.nel)
println("Number of vertices: ", mesh.nn)
println("Number of p nodes:  ", mesh.np)

field  = rand(mesh.nel) # force to make a random field, if field is constant Makie crashes!!
PlotMakie(mesh, field, xmin, xmax, ymin, ymax; cmap=:turbo)
