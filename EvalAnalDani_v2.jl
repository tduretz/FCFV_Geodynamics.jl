import Plots
using LoopVectorization

################################################

@views function SolutionFields_p(mm, mc, rc, er, gr, x, y, in)
    # Outside
    pmat   = (4*mm*rc^2*(mm - mc)*(er*x^2 + gr*x*y - er*y^2))/((x^2 + y^2)^2*(mm + mc))
    # Inside
    pinc   = 0.0
    # Where it's needed
    p      = (in==0)*pmat + (in==1)*pinc 
    return p 
end

################################################

@views function SolutionFields_v(mm, mc, rc, er, gr, x, y, in)
    # Outside
    A = mm*(mc-mm)/(mc+mm);
    v1mat = (2*A*er*rc^4*x^3 + 3*A*gr*rc^4*x^2*y - 6*A*er*rc^4*x*y^2 - A*gr*rc^4*y^3 - 4*A*er*rc^2*x^5 - 
        4*A*gr*rc^2*x^4*y - 4*A*gr*rc^2*x^2*y^3 + 4*A*er*rc^2*x*y^4 + 2*er*mm*x^7 + 2*gr*mm*x^6*y + 
        6*er*mm*x^5*y^2 + 6*gr*mm*x^4*y^3 + 6*er*mm*x^3*y^4 + 6*gr*mm*x^2*y^5 + 2*er*mm*x*y^6 + 
        2*gr*mm*y^7)/(2*mm*(x^2 + y^2)^3)
    v2mat = -(A*gr*rc^4*x^3 - 6*A*er*rc^4*x^2*y - 3*A*gr*rc^4*x*y^2 + 2*A*er*rc^4*y^3 + 4*A*er*rc^2*x^4*y + 
        4*A*gr*rc^2*x^3*y^2 + 4*A*gr*rc^2*x*y^4 - 4*A*er*rc^2*y^5 + 2*er*mm*x^6*y + 6*er*mm*x^4*y^3 + 
        6*er*mm*x^2*y^5 + 2*er*mm*y^7)/(2*mm*(x^2 + y^2)^3)
    # Inside
    v1inc  = (4*er*mm*x + 3*gr*mm*y + gr*mc*y)/(2*(mm + mc));
    v2inc  = -(4*er*mm*y - gr*mm*x + gr*mc*x)/(2*(mm + mc));
    # Where it's needed
    vx      = (in==0)*v1mat + (in==1)*v1inc  
    vy      = (in==0)*v2mat + (in==1)*v2inc  
    return vx, vy 
end

################################################

@views function SolutionFields_dv(mm, mc, rc, er, gr, x, y, in)
    # Outside
    A = mm*(mc-mm)/(mc+mm);
    dvxdxmat = (- 3*A*er*rc^4*x^4 - 6*A*gr*rc^4*x^3*y + 18*A*er*rc^4*x^2*y^2 + 6*A*gr*rc^4*x*y^3 - 3*A*er*rc^4*y^4 + 2*A*er*rc^2*x^6 + 4*A*gr*rc^2*x^5*y - 10*A*er*rc^2*x^4*y^2 - 10*A*er*rc^2*x^2*y^4 - 4*A*gr*rc^2*x*y^5 + 2*A*er*rc^2*y^6 + er*mm*x^8 + 4*er*mm*x^6*y^2 + 6*er*mm*x^4*y^4 + 4*er*mm*x^2*y^6 + er*mm*y^8)/(mm*(x^2 + y^2)^4);
    dvxdymat = (3*A*gr*rc^4*x^4 - 24*A*er*rc^4*x^3*y - 18*A*gr*rc^4*x^2*y^2 + 24*A*er*rc^4*x*y^3 + 3*A*gr*rc^4*y^4 - 4*A*gr*rc^2*x^6 + 24*A*er*rc^2*x^5*y + 8*A*gr*rc^2*x^4*y^2 + 16*A*er*rc^2*x^3*y^3 + 12*A*gr*rc^2*x^2*y^4 - 8*A*er*rc^2*x*y^5 + 2*gr*mm*x^8 + 8*gr*mm*x^6*y^2 + 12*gr*mm*x^4*y^4 + 8*gr*mm*x^2*y^6 + 2*gr*mm*y^8)/(2*mm*(x^2 + y^2)^4);
    dvydxmat = (A*rc^2*(3*gr*rc^2*x^4 - 24*er*rc^2*x^3*y - 18*gr*rc^2*x^2*y^2 + 24*er*rc^2*x*y^3 + 3*gr*rc^2*y^4 + 8*er*x^5*y + 12*gr*x^4*y^2 - 16*er*x^3*y^3 + 8*gr*x^2*y^4 - 24*er*x*y^5 - 4*gr*y^6))/(2*mm*(x^2 + y^2)^4);
    dvydymat = -(- 3*A*er*rc^4*x^4 - 6*A*gr*rc^4*x^3*y + 18*A*er*rc^4*x^2*y^2 + 6*A*gr*rc^4*x*y^3 - 3*A*er*rc^4*y^4 + 2*A*er*rc^2*x^6 + 4*A*gr*rc^2*x^5*y - 10*A*er*rc^2*x^4*y^2 - 10*A*er*rc^2*x^2*y^4 - 4*A*gr*rc^2*x*y^5 + 2*A*er*rc^2*y^6 + er*mm*x^8 + 4*er*mm*x^6*y^2 + 6*er*mm*x^4*y^4 + 4*er*mm*x^2*y^6 + er*mm*y^8)/(mm*(x^2 + y^2)^4);
    # Inside
    dvxdxinc = (2*er*mm)/(mm + mc);
    dvxdyinc = (gr*(3*mm + mc))/(2*(mm + mc));
    dvydxinc = (gr*(mm - mc))/(2*(mm + mc));
    dvydyinc = -(2*er*mm)/(mm + mc);
    # Where it's needed
    dvxdx      = (in==0)*dvxdxmat + (in==1)*dvxdxinc  
    dvxdy      = (in==0)*dvxdymat + (in==1)*dvxdyinc 
    dvydx      = (in==0)*dvydxmat + (in==1)*dvydxinc  
    dvydy      = (in==0)*dvydymat + (in==1)*dvydyinc 
    return dvxdx, dvxdy, dvydx, dvydy
end

################################################

function Evaluation2D!( ncx, ncy, xc, yc, p, vx, vy, rc, mm, mc )
    gr  = 0;                        # Simple shear: gr=1, er=0
    er  = -1;                       # Strain rate
    @tturbo for i=1:ncx
        for j=1:ncy
            x, y                           = xc[i], yc[j]
            in                             = sqrt(x^2.0 + y^2.0)<=rc
            pa                             = SolutionFields_p( mm, mc, rc, er, gr, x, y, in)
            vxa, vya                       = SolutionFields_v( mm, mc, rc, er, gr, x, y, in)
            dvxdxa, dvxdya, dvydxa, dvydya = SolutionFields_dv(mm, mc, rc, er, gr, x, y, in)
            p[i,j], vx[i,j], vy[i,j]       = pa, vxa, vya
        end
    end
end

################################################

@views function main()

    # Create domain
    xmin = -3.0
    xmax = 3.0 
    ymin = -3.0
    ymax = 3.0
    ncx  = 50
    ncy  = 40
    xc   = LinRange(xmin,xmax,ncx)
    yc   = LinRange(ymin,ymax,ncy)
    xc2d = repeat(xc, 1, length(yc))
    yc2d = repeat(yc, 1, length(xc))'
    p    = zeros(ncx,ncy)
    vx   = zeros(ncx,ncy)
    vy   = zeros(ncx,ncy)
    rc   = 1.0
    mm   = 1.0
    mc   = 100.0

    for itest=1:5
        @time Evaluation2D!( ncx, ncy, xc, yc, p, vx, vy, rc, mm, mc )
    end

    p2 = Plots.heatmap(xc, yc, p', c=:jet1 )
    display(Plots.plot(p2))

end

################################################

#main()