# FCFV_geodynamics.jl

# Step 0: The script:[`VisualisationTriangles.jl`](VisualisationTriangles.jl).

## Step 0.1: Creating a triangular mesh with Triangle via TriangleMesh.

## Step 0.2: Computation of a field on the triangles.

Some interesting timings:
Computation of the field on the triangles on the low resolution mesh using either a vectorized loop (@avx - LoopVectorization package) or standard vectorisation.

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

At high resolution:

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

As a conclusion, the package LoopVectorization rocks!

## Step 0.3: Visualising data ona Triangular mesh using Pyplot

It was difficult to find a straighforward way to visualise data on triangle within existing Julia packages. Luckily, matplotlib has everything neede and is readily avalable via the PyPlot package.

Low resolution:
![](/images/0_Low_res.png)

High resolution:
![](/images/0_high_res.png)

