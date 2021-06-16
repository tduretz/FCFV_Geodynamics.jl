function EvalAnalDani( x, z, rc, mm, mc )

# x  = -1.0
# z  = -1.0
# rc = 5
# mc = 10

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
# PRESSURE CALCULATION OUTSIDE OF AN INCLUSION IN THE Z-PLANE
# --------------------------------------------------------------

# INSIDE CLAST
if sqrt(x^2 + z^2)<=rc
    
    Z       =   x + i*z;
    P       =   0;   # if you want you can add NaN, but according to Schmid's thesis it's zero inside
    
    # VELOCITY
    V_tot      =  (mm/(mc+mm))*(i*gr+2*er)*conj(Z)-(i/2)*gr*Z;
    vx         =  real(V_tot);
    vz         =  imag(V_tot);
    eta        =  mc;
#     tauc           =  2*sqrt(4*er^2 + gr^2) * (mm*mc) /(mc+mm);
    
    # Evaluate stresses
    d_phi_z        = -i/2*mc*gr;
    d_d_phi_z_z    = 0;
    d_psi_z        = 2*(i*gr-2*er)*(mc*mm)/(mc+mm);
    szz        = 2*real(d_phi_z) + real( (x - i*z) * d_d_phi_z_z + d_psi_z);
    sxx        = 4*real(d_phi_z) - szz;
    sxz        = imag( (x - i*z) * d_d_phi_z_z + d_psi_z); 
else
    # OUTSIDE CLAST, RESP. MATRIX
    Z              =   x + i*z;
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
    vz         =  imag(V_tot);
    eta        = mm;
      
    # Evaluate stresses (Valid one)
    d_d_phi_z_z    = -(i*gr+2*er)*A*rc^2/Z^3;
    d_psi_z        = (i*gr-2*er)*mm + (i*gr+2*er)*A*rc^4*Z^(-4);
    szz        = 2*real(d_phi_z) + real( (x - i*z) * d_d_phi_z_z + d_psi_z);
    sxx        = 4*real(d_phi_z) -  szz;
    sxz        = imag(  (x - i*z) .* d_d_phi_z_z + d_psi_z );
    
end

return vx, vz, P, sxx, szz, sxz
end
