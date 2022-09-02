using CellArrays, StaticArrays, Setfield

function main_CellArrays()
nel      = 2
nip      = 1
celldims = (3, 3)
Cell     = SMatrix{celldims..., Float64, prod(celldims)}
D        = CPUCellArray{Cell}(undef, nel, nip) 
D.data  .= 0.0

celldims = (3, 1)
Cell     = SMatrix{celldims..., Float64, prod(celldims)}
x        = CPUCellArray{Cell}(undef, nel, nip)
x.data  .= 1

D .* x

for ip=1:nip
    for e=1:nel
        for i=1:celldims[1]
            field(D,i,i)[e,ip] = 2.0
        end
    end
end

De = [2 0 0; 0 2 0; 0 0 2]

for e=1:nel
    field(D,:,:)[e,ip] .= De[:]
end

@show D

end

main_CellArrays()
