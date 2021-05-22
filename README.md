# FCFV_geodynamics.jl

This repository contains an implementation of the Face-Centered Finite Volume (FCFV) Poisson solver presented in Sevilla et al. (2018). The code heavily relies on the [LoopVectorization package](https://github.com/JuliaSIMD/LoopVectorization.jl). The code supports quadrangular and triangular elements. Triangular mesh generation relies on the [TriangleMesh package](https://github.com/konsim83/TriangleMesh.jl), which is a Julia wrapper to [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) (Shewchuk, 2002). 

Convergence to manufactured solution (see Sevilla et al., 2018):<br/>

![](/images/1_conv_diff_cst.png)

Total execution time of Julia code, as fucntion of the total number of dofs (excluding visualisation):<br/>

![](/images/1_time_diff_cst.png)

An example of computation on quads (512^2 - 262144 elements):

![](/images/1_quad_diff_cst.png)

An example of computation on triangles (450219 elements):

![](/images/1_tri_diff_cst.png)

The "grainy" pattern is due to the white outlines around elments which I did not manage to remove when plotting with Makie.

# References

Sevilla, R, Giacomini, M, Huerta, A. A face-centred finite volume method for second-order elliptic problems. Int J Numer Methods Eng. 2018; 115: 986– 1014. https://doi.org/10.1002/nme.5833

Jonathan Richard Shewchuk, Delaunay Refinement Algorithms for Triangular Mesh Generation, Computational Geometry: Theory and Applications 22(1-3):21-74, May 2002
