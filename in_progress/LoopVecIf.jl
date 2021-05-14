using LoopVectorization

nel  = 100

# Standard loop
res1 = zeros(nel)
for iel=1:nel
    dot = sin(iel/nel*pi) - 0.5
    if dot<0.0
        res1[iel] = 1
    else
        res1[iel] = 2
    end
end

# @avx vectorized
res2 = zeros(nel)
@avx for iel=1:nel
    dot = sin(iel/nel*pi) - 0.5
    res2[iel] = (dot<0.0)*1.0 + (dot>0.0)*2.0
end

println("Sum res1 = ", sum(res1))
println("Sum res2 = ", sum(res2))