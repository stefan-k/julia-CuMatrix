include("cuda.jl")

# The CuMatrix class
type CuMatrix
    T::Type
    ptr::Ptr{Void}
    dims::(Integer,Integer)

    # Construct Matrix on device
    function CuMatrix(T::Type, dims::(Integer, Integer))
        if T != Float32 && T != Float64
            error("No integer or boolean support yet")
        end
        ptr = jl_cuda_malloc(T, dims)
        new(T, ptr, dims)
    end

    # Copy Matrix from host
    function CuMatrix(in::Matrix)
        T = eltype(in)
        dims = size(in)
        ptr = jl_cuda_malloc(T, dims)
        jl_mem_device(ptr, in)
        new(T, ptr, dims)
    end

    function CuMatrix(T::Type, ptr::Ptr{Void}, dims::(Integer, Integer))
        new(T, ptr, dims)
    end

    # Aliasing
    CuMatrix(T::Type, rows::Integer, cols::Integer) = CuMatrix(T, (rows, cols))
    CuMatrix(T::Type, rows::Integer) = CuMatrix(T, (rows, 1))
end

# Get matrix from device to host
function Array(in::CuMatrix)
    out = Array(in.T, in.dims[1], in.dims[2])
    jl_mem_host(out, in.ptr)
    return out
end

# Perform a deep copy
function copy(in::CuMatrix)
    ptr = jl_cuda_malloc(in.T, in.dims)
    bytes::Int32 = in.dims[1] * in.dims[2] * sizeof(in.T)
    jl_mem_copy(ptr, in.ptr, bytes)
    CuMatrix(in.T, ptr, in.dims)
end
    
# Display function
function print(in::CuMatrix)
    print("On GPU\n")
    print(Array(in), "\n")
end
