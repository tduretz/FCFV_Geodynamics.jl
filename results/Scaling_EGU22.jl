using Plots

# QUADS
q_nit  = [150, 300,  500,   1000,  1900];
q_ndof = [840, 3280, 12960, 51520, 205440];
q_nx   = [20,  40,   80,    160,   320];

# TRIANGLES
t_nit  = [300,  600,   1050,  2050,   4000];
t_ndof = [1019, 155,   16502, 65733, 262913];
t_nx   = [20,   40,    80,    160,    320];

p1 = plot(  q_ndof, q_nit, label="Quads", markershape=:square )
p1 = plot!( t_ndof, t_nit, label="Triangles", markershape=:dtriangle, xlabel="dofs", ylabel="Iters", foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomright )
display(p1)

p2 = plot(  q_ndof, q_nit./q_nx, label="Quads", markershape=:square )
p2 = plot!( t_ndof, t_nit./t_nx, label="Triangles", markershape=:dtriangle, xlabel="dofs", ylabel="Iters/nx", foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomright  )
display(plot(p1,p2, layout = (2, 1)))

png(string(@__DIR__,"/Scaling_PT"))