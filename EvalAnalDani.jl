function EvalAnalDani( x, y, rc, mm, mc )
# ---------------------------------------------------------------------------
# ANALYTICAL SOLUTION - PRESSURE AND VELOCITY AROUND A CIRCULAR INCLUSION:
#
# BASED ON DANI SCHMID'S 2002 CYL_P_MATRIX.M
# FAR FIELD FLOW - VISCOSITIES - GEOMETRY
#
# ---------------------------------------------------------------------------

# INPUT:
# gr  = 0;                        # Simple shear: gr=1, er=0
# er  = -1;                       # Strain rate
gr  = 1; 
er  = 0;
A   =   mm*(mc-mm)/(mc+mm);
i   =   1im;

# --------------------------------------------------------------
# PRESSURE CALCULATION OUTSIDE OF AN INCLUSION IN THE y-PLANE
# --------------------------------------------------------------

# println(x)
# println(y)
# println(rc)


# INSIDE CLAST
if sqrt(x^2.0 + y^2.0)<=rc
    
    Z       =   x + i*y;
    P       =   0;  
    
    # VELOCITY
    V_tot      =  (mm/(mc+mm))*(i*gr+2*er)*conj(Z)-(i/2)*gr*Z;
    vx         =  real(V_tot);
    vy         =  imag(V_tot);
    eta        =  mc;

    A   = mm*(mc-mm)/(mc+mm);
    v1x = (- 3.0*A*er*rc^4*x^4 - 6.0*A*gr*rc^4*x^3.0*y + 18.0*A*er*rc^4*x^2.0*y^2.0 + 6.0*A*gr*rc^4*x*y^3.0 - 3.0*A*er*rc^4*y^4 + 2.0*A*er*rc^2.0*x^6.0 + 4*A*gr*rc^2.0*x^5*y - 10*A*er*rc^2.0*x^4*y^2.0 - 10*A*er*rc^2.0*x^2.0*y^4 - 4*A*gr*rc^2.0*x*y^5 + 2.0*A*er*rc^2.0*y^6.0 + er*mm*x^8 + 4*er*mm*x^6.0*y^2.0 + 6.0*er*mm*x^4*y^4 + 4*er*mm*x^2.0*y^6.0 + er*mm*y^8)/(mm*(x^2.0 + y^2.0)^4);
    v1y = (3.0*A*gr*rc^4*x^4 - 24.0*A*er*rc^4*x^3.0*y - 18.0*A*gr*rc^4*x^2.0*y^2.0 + 24.0*A*er*rc^4*x*y^3.0 + 3.0*A*gr*rc^4*y^4 - 4*A*gr*rc^2.0*x^6.0 + 24.0*A*er*rc^2.0*x^5*y + 8*A*gr*rc^2.0*x^4*y^2.0 + 16*A*er*rc^2.0*x^3.0*y^3.0 + 12*A*gr*rc^2.0*x^2.0*y^4 - 8*A*er*rc^2.0*x*y^5 + 2.0*gr*mm*x^8 + 8*gr*mm*x^6.0*y^2.0 + 12*gr*mm*x^4*y^4 + 8*gr*mm*x^2.0*y^6.0 + 2.0*gr*mm*y^8)/(2.0*mm*(x^2.0 + y^2.0)^4);
    v2x = (A*rc^2.0*(3.0*gr*rc^2.0*x^4 - 24.0*er*rc^2.0*x^3.0*y - 18.0*gr*rc^2.0*x^2.0*y^2.0 + 24.0*er*rc^2.0*x*y^3.0 + 3.0*gr*rc^2.0*y^4 + 8*er*x^5*y + 12*gr*x^4*y^2.0 - 16*er*x^3.0*y^3.0 + 8*gr*x^2.0*y^4 - 24.0*er*x*y^5 - 4*gr*y^6.0))/(2.0*mm*(x^2.0 + y^2.0)^4);
    v2y = -(- 3.0*A*er*rc^4*x^4 - 6.0*A*gr*rc^4*x^3.0*y + 18.0*A*er*rc^4*x^2.0*y^2.0 + 6.0*A*gr*rc^4*x*y^3.0 - 3.0*A*er*rc^4*y^4 + 2.0*A*er*rc^2.0*x^6.0 + 4*A*gr*rc^2.0*x^5*y - 10*A*er*rc^2.0*x^4*y^2.0 - 10*A*er*rc^2.0*x^2.0*y^4 - 4*A*gr*rc^2.0*x*y^5 + 2.0*A*er*rc^2.0*y^6.0 + er*mm*x^8 + 4*er*mm*x^6.0*y^2.0 + 6.0*er*mm*x^4*y^4 + 4*er*mm*x^2.0*y^6.0 + er*mm*y^8)/(mm*(x^2.0 + y^2.0)^4);
    Txy = mm*(v1y + v2x);
    Txx = mm*(v1x + v1x);
    Tyy = mm*(v2y + v2y);
    sxx = real(-P + Txx)
    syy = real(-P + Tyy)
    sxy = real(Txy)
else
     # OUTSIDE CLAST, RESP. MATRIX
     Z              =   x + i*y;
     # PRESSURE
     P          =   -2.0*mm.*(mc-mm)./(mc+mm).*real(rc^2.0/Z.^2.0*(i*gr+2*er));
     
     # VELOCITY
     phi_z          = -(i/2)*mm*gr*Z-(i*gr+2*er)*A*rc^2*Z^(-1);
     d_phi_z        = -(i/2)*mm*gr + (i*gr+2*er)*A*rc^2/Z^2;
     conj_d_phi_z   = conj(d_phi_z);
     psi_z          = (i*gr-2*er)*mm*Z-(i*gr+2*er)*A*rc^4*Z^(-3);
     conj_psi_z     = conj(psi_z);
     
     V_tot          = (phi_z- Z*conj_d_phi_z - conj_psi_z) / (2*mm);
     vx         =  real(V_tot);
     vy         =  imag(V_tot);
     eta        = mm;
      
    # Evaluate stresses (Valid one)
    v1x = (2.0*er*mm)/(mm + mc);
    v1y = (gr*(3.0*mm + mc))/(2.0*(mm + mc));
    v2x = (gr*(mm - mc))/(2.0*(mm + mc));
    v2y = -(2.0*er*mm)/(mm + mc);
    Txy = mc*(v1y + v2x);
    Txx = mc*(v1x + v1x);
    Tyy = mc*(v2y + v2y);
    sxx = -P + Txx
    syy = -P + Tyy
    sxy = Txy
end

return vx, vy, P, sxx, syy, sxy, v1x, v1y, v2x, v2y, eta
end


function Tractions( x, y, rc, mm, mc, phase )

# ---------------------------------------------------------------------------
# ANALYTICAL SOLUTION - PRESSURE AND VELOCITY AROUND A CIRCULAR INCLUSION:
#
# BASED ON DANI SCHMID'S 2002 CYL_P_MATRIX.M
# FAR FIELD FLOW - VISCOSITIES - GEOMETRY
#
# ---------------------------------------------------------------------------

# INPUT:
gr  = 0;                        # Simple shear: gr=1, er=0
er  = -1;                       # Strain rate
A   =   mm*(mc-mm)/(mc+mm);
i   =   1im;

# --------------------------------------------------------------
# PRESSURE CALCULATION OUTSIDE OF AN INCLUSION IN THE y-PLANE
# --------------------------------------------------------------

# INSIDE CLAST
if phase == 2
    
    Z       =   x + i*y;
    P       =   0;   # if you want you can add NaN, but according to Schmid's thesis it's zero inside
    
    # VELOCITY
    V_tot      =  (mm/(mc+mm))*(i*gr+2*er)*conj(Z)-(i/2)*gr*Z;
    vx         =  real(V_tot);
    vy         =  imag(V_tot);
    eta        =  mc;
    
    # Evaluate stresses
    v1x = (2*er*mm)/(mm + mc);
    v1y = (gr*(3*mm + mc))/(2*(mm + mc));
    v2x = (gr*(mm - mc))/(2*(mm + mc));
    v2y = -(2*er*mm)/(mm + mc);

    Txy = mm*(v1y + v2x);
    Txx = mm*(v1x + v1x);
    Tyy = mm*(v2y + v2y);
    sxx = real(-P + Txx)
    syy = real(-P + Tyy)
    sxy = real(Txy)
end
if phase == 1
    # OUTSIDE CLAST, RESP. MATRIX
    Z              =   x + i*y;
    # PRESSURE
    P          =   -2.0*mm.*(mc-mm)./(mc+mm).*real(rc^2.0/Z.^2.0*(i*gr+2*er));
    
    # VELOCITY
    phi_z          = -(i/2)*mm*gr*Z-(i*gr+2*er)*A*rc^2*Z^(-1);
    d_phi_z        = -(i/2)*mm*gr + (i*gr+2*er)*A*rc^2/Z^2;
    conj_d_phi_z   = conj(d_phi_z);
    psi_z          = (i*gr-2*er)*mm*Z-(i*gr+2*er)*A*rc^4*Z^(-3);
    conj_psi_z     = conj(psi_z);
    
    V_tot          = (phi_z- Z*conj_d_phi_z - conj_psi_z) / (2*mm);
    vx         =  real(V_tot);
    vy         =  imag(V_tot);
    eta        = mm;
      
    # Evaluate stresses (Valid one)
    A = mm*(mc-mm)/(mc+mm);
    v1x = (- 3*A*er*rc^4*x^4 - 6*A*gr*rc^4*x^3*y + 18*A*er*rc^4*x^2*y^2 + 6*A*gr*rc^4*x*y^3 - 3*A*er*rc^4*y^4 + 2*A*er*rc^2*x^6 + 4*A*gr*rc^2*x^5*y - 10*A*er*rc^2*x^4*y^2 - 10*A*er*rc^2*x^2*y^4 - 4*A*gr*rc^2*x*y^5 + 2*A*er*rc^2*y^6 + er*mm*x^8 + 4*er*mm*x^6*y^2 + 6*er*mm*x^4*y^4 + 4*er*mm*x^2*y^6 + er*mm*y^8)/(mm*(x^2 + y^2)^4);
    v1y = (3*A*gr*rc^4*x^4 - 24*A*er*rc^4*x^3*y - 18*A*gr*rc^4*x^2*y^2 + 24*A*er*rc^4*x*y^3 + 3*A*gr*rc^4*y^4 - 4*A*gr*rc^2*x^6 + 24*A*er*rc^2*x^5*y + 8*A*gr*rc^2*x^4*y^2 + 16*A*er*rc^2*x^3*y^3 + 12*A*gr*rc^2*x^2*y^4 - 8*A*er*rc^2*x*y^5 + 2*gr*mm*x^8 + 8*gr*mm*x^6*y^2 + 12*gr*mm*x^4*y^4 + 8*gr*mm*x^2*y^6 + 2*gr*mm*y^8)/(2*mm*(x^2 + y^2)^4);
    v2x = (A*rc^2*(3*gr*rc^2*x^4 - 24*er*rc^2*x^3*y - 18*gr*rc^2*x^2*y^2 + 24*er*rc^2*x*y^3 + 3*gr*rc^2*y^4 + 8*er*x^5*y + 12*gr*x^4*y^2 - 16*er*x^3*y^3 + 8*gr*x^2*y^4 - 24*er*x*y^5 - 4*gr*y^6))/(2*mm*(x^2 + y^2)^4);
    v2y = -(- 3*A*er*rc^4*x^4 - 6*A*gr*rc^4*x^3*y + 18*A*er*rc^4*x^2*y^2 + 6*A*gr*rc^4*x*y^3 - 3*A*er*rc^4*y^4 + 2*A*er*rc^2*x^6 + 4*A*gr*rc^2*x^5*y - 10*A*er*rc^2*x^4*y^2 - 10*A*er*rc^2*x^2*y^4 - 4*A*gr*rc^2*x*y^5 + 2*A*er*rc^2*y^6 + er*mm*x^8 + 4*er*mm*x^6*y^2 + 6*er*mm*x^4*y^4 + 4*er*mm*x^2*y^6 + er*mm*y^8)/(mm*(x^2 + y^2)^4);
    Txy = mc*(v1y + v2x);
    Txx = mc*(v1x + v1x);
    Tyy = mc*(v2y + v2y);
    sxx = -P + Txx
    syy = -P + Tyy
    sxy = Txy
end

return P, real(v1x), real(v1y), real(v2x), real(v2y)
end
