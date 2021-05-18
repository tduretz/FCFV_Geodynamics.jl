ENV["MPLBACKEND"]="Qt5Agg"
include("CreateMesh.jl")
using LoopVectorization
using PyPlot
pygui(true)

function tplot(p, t, v)
    # Plot triangular mesh with nodes `p` and triangles `t`
    clf()
    tris = convert(Array{Int64}, hcat(t...)')
    # fig1, ax1 = plt.subplots()
    display(tripcolor(first.(p), last.(p), tris .- 1, v,
              cmap="viridis", edgecolors="none", linewidth=0))
    axis("equal")
    ylim([0, 1])
    xlim([0, 1])
    title("Low res.")
    xlabel("x")
    ylabel("y")
    colorbar()
    show()
    return 
end

function qplot(x, y, v)
    clf()
    display( pcolor(x, y, v) )
    colorbar()
    xlabel("x")
    ylabel("y")
    show()
end

function Tanalytic2!( mesh, xc , yc, T, a, b, c, d, alp, bet )
    if mesh.type == "triangle" nn_el = 3.0; nn_el_int=3; end
    if mesh.type == "quad"     nn_el = 4.0; nn_el_int=4; end
    # Evaluate T analytic on barycentres
    for j=1:mesh.nn_el # as fast as above and less cryptic
        @. xc = 0.0
        @. yc = 0.0
        @. xc = xc + 1.0/mesh.nn_el * mesh.xn[mesh.e2n[j,:]] 
        @. yc = yc + 1.0/mesh.nn_el * mesh.yn[mesh.e2n[j,:]] 
    end
    @. T = exp(alp*sin(a*xc + c*yc) + bet*cos(b*xc + d*yc));
    # xn = mesh.xn
    # yn = mesh.yn
    # e2n = mesh.e2n
    #  for i=1:mesh.nel
    #     xc[i] = 0;
    #     yc[i] = 0;
    #     @avx for j=1:3
    #         # xc[i] += 1.0/nn_el * mesh.xn[mesh.e2n[j,i]] 
    #         # yc[i] += 1.0/nn_el * mesh.yn[mesh.e2n[j,i]] 
    #         xc[i] += 1.0/nn_el * xn[e2n[j,i]] 
    #         yc[i] += 1.0/nn_el * yn[e2n[j,i]] 
    #     end
    # end
    # for i=1:mesh.nel
    #     T[i] = exp(alp.sin(a*xc[i] + c*yc[i]) + bet*cos(b*xc[i] + d*yc[i]));
    # end
return
end

function main()

    # Create sides of mesh
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny     = 20, 20
    quad = false

    if quad==true 
        mesh = MakeQuadMesh( nx, ny, xmin, xmax, ymin, ymax )
    elseif quad==false  
        mesh = MakeTriangleMesh( nx, ny, xmin, xmax, ymin, ymax ) 
    end

    println(mesh.nel)

    # Generate function to be visualised on mesh
    alp = 0.1; bet = 0.3; a = 5.1; b = 4.3; c = -6.2; d = 3.4;
    xc = zeros(mesh.nel)
    yc = zeros(mesh.nel)
    T  = zeros(mesh.nel)
    ncalls = 4
    # A) Loop version with @avx
    # @printf("Looped, %d times:\n", ncalls)
    for icall=1:ncalls
        @time Tanalytic2!(mesh, xc ,yc , T, a, b, c, d, alp, bet)
    end

    if mesh.type == "triangle"  tplot(mesh.xyviz, mesh.e2nviz,                    T ) end
    if mesh.type == "quad"      qplot(mesh.xcviz, mesh.ycviz,  reshape(T, (nx,ny) ) ) end 

end

main()