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

    # These are local mat-vec products using cell arrays - great
    D .* x

    #-----------------------------------------------
    nel      = 2
    celldims = (3, 3)
    Cell     = SMatrix{celldims..., Float64, prod(celldims)}
    C        = CPUCellArray{Cell}(undef, nel) 
    C.data  .= 0.0

    # The aims is to change te values of each 3x3 matrix
    # 1 - with a double loop for each element: it works
    for e=1:nel
        for i=1:celldims[1]
            for j=1:celldims[2]
                field(C,i,j)[e] = 1.0
            end
        end
    end
    @show C

    # 2 - with a broadcast: it doesn't work so far
    for e=1:nel
        field(C,:,:)[e] .= 1.0
    end
    @show C

end

main_CellArrays()
