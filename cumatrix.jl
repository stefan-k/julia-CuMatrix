include("cuda.jl")

type CuMatrix
    T::Type
    ptr::Ptr{Void}
    dims::(Integer,Integer)
    function CuMatrix(T::Type, rows::Integer, cols::Integer)
        dims = (rows, cols)
        ptr = jl_cuda_malloc(T, rows, cols)
        new(T, ptr, dims)
    end
end
