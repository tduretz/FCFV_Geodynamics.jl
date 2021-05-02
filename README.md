# FCFV_geodynamics.jl

# Step 0: The script:[`VisualisationTriangles.jl`](VisualisationTriangles.jl).

## Step 0.1: Creating a triangular mesh with Triangle via TriangleMesh.

The [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) mesh generator is accessible through several packages. It is not clear which one should be used (?). Here, I've chosen to use [TriangleMesh](https://github.com/konsim83/TriangleMesh.jl) which allows to parse in command line arguments of the original Triangle.  

## Step 0.2: Computation of a field on the triangles.

Some interesting timings:
Computation of the field on the triangles on a low resolution mesh (max triangle area 0.01) using either a vectorized loop (@avx - [LoopVectorization package](https://github.com/JuliaSIMD/LoopVectorization.jl)) or standard vectorisation.

Looped, 4 times:<br/>
  0.000015 seconds<br/>
  0.000006 seconds<br/>
  0.000005 seconds<br/>
  0.000005 seconds<br/>
Vectorized, 4 times:<br/>
  0.000016 seconds (12 allocations: 19.125 KiB)<br/>
  0.000013 seconds (12 allocations: 19.125 KiB)<br/>
  0.000015 seconds (12 allocations: 19.125 KiB)<br/>
  0.000014 seconds (12 allocations: 19.125 KiB)<br/>

At high resolution (max triangle area 0.0001):

Looped, 4 times:<br/>
  0.005380 seconds<br/>
  0.005565 seconds<br/>
  0.005576 seconds<br/>
  0.005361 seconds<br/>
Vectorized, 4 times:<br/>
  0.019515 seconds (24 allocations: 15.630 MiB)<br/>
  0.029903 seconds (24 allocations: 15.630 MiB, 28.99% gc time)<br/>
  0.015696 seconds (24 allocations: 15.630 MiB)<br/>
  0.015948 seconds (24 allocations: 15.630 MiB)<br/>

As a conclusion, the package [LoopVectorization](https://github.com/JuliaSIMD/LoopVectorization.jl) rocks!

## Step 0.3: Visualising data on a Triangular mesh using Pyplot

It was difficult to find a straighforward way to visualise data on triangles within existing Julia packages (docs are sometimes incomplete or practical examples are simply missing). Luckily, matplotlib has everything needed and is readily avalable via the PyPlot package.

Low resolution:<br/>

![](/images/0_LowRes.png)

High resolution:<br/>

![](/images/0_HighRes.png)

