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
        ptr = cuda_malloc(T, dims)
        Res = new(T, ptr, dims)
        finalizer(Res, CuFree)
        Res
    end

    # Copy Matrix from host
    function CuMatrix(in::Matrix)
        T = eltype(in)
        dims = size(in)
        ptr = cuda_malloc(T, dims)
        mem_device(ptr, in)
        Res = new(T, ptr, dims)
        finalizer(Res, CuFree)
        Res
    end

    # Constructor with existing device pointer
    function CuMatrix(T::Type, ptr::Ptr{Void}, dims::(Integer, Integer))
        Res = new(T, ptr, dims)
        finalizer(Res, CuFree)
        Res
    end

    # Aliasing
    CuMatrix(T::Type, rows::Integer, cols::Integer) = CuMatrix(T, (rows, cols))
    CuMatrix(T::Type, rows::Integer) = CuMatrix(T, (rows, 1))

    # Default type: Single precision (for compatibility with older cards)
    CuMatrix(rows::Integer, cols::Integer) = CuMatrix(Float32, rows, cols)
    CuMatrix(rows::Integer) = CuMatrix(Float32, rows, 1)
end

# Get matrix from device to host
function Array(in::CuMatrix)
    out = Array(in.T, in.dims[1], in.dims[2])
    mem_host(out, in.ptr)
    return out
end

# Perform a deep copy
function copy(in::CuMatrix)
    ptr = cuda_malloc(in.T, in.dims)
    bytes::Int32 = in.dims[1] * in.dims[2] * sizeof(in.T)
    mem_copy(ptr, in.ptr, bytes)
    CuMatrix(in.T, ptr, in.dims)
end
    
# Display function
function print(in::CuMatrix)
    print("On GPU\n")
    print(Array(in), "\n")
end

# Freeing memory
function CuFree(in::CuMatrix)
    cuda_free(in.ptr)
end

# Random Number Generators
function curand(T::Type, rows::Integer, cols::Integer)
    out = CuMatrix(T, rows, cols)
    cuda_rand(T, out.ptr, rows * cols)
    return out
end

function curandn(T::Type, rows::Integer, cols::Integer)
    # curand only supports randn for even number of elements
    count = rows * cols
    count += count & 1 # Increase the malloc size
    ptr = cuda_malloc(T, count)
    cuda_randn(T, ptr, count)
    # But only use the required number of elements
    CuMatrix(T, ptr, (rows, cols))
end

curand(rows::Integer, cols::Integer) = curand(Float32, rows, cols)
curandn(rows::Integer, cols::Integer) = curandn(Float32, rows, cols)

# BLAS Functions
function (*)(A::CuMatrix, B::CuMatrix)
    if (A.dims[2] != B.dims[1])
        error("Inner dimension mismatch in Matrix multiply")
    end
    if (A.T != B.T)
        error("Precision mismatch in Matrix multiply")
    end
    C = CuMatrix(A.T, (A.dims[1], B.dims[2]))

    m = convert(Int32, C.dims[1])
    n = convert(Int32, C.dims[2])
    k = convert(Int32, B.dims[1])
    
    cuda_gemm('N', 'N', m, n, k,
            one(A.T), A.ptr, m, B.ptr, k,
            zero(A.T), C.ptr, m)
    return C
end

function cuamax(A::CuMatrix)
    result = CuMatrix(Int32, 1, 1)
    # needs some more checking
    n = convert(Int32, A.dims[1] > A.dims[2] ? A.dims[1] : A.dims[2])
    cuda_amax(n,  A.ptr, int32(1), result.ptr)
    return result
end

function cuasum(A::CuMatrix)
    result = CuMatrix(A.T, 1, 1)
    # needs some more checking
    n = convert(Int32, A.dims[1] > A.dims[2] ? A.dims[1] : A.dims[2])
    cuda_asum(n,  A.ptr, int32(1), result.ptr)
    return result
end

function cuscal(A::CuMatrix, alpha)
    # needs some more checking
    n = convert(Int32, A.dims[1] > A.dims[2] ? A.dims[1] : A.dims[2])
    cuda_scal(n, alpha, A.ptr, int32(1))
end

function cudot(A::CuMatrix, B::CuMatrix)
    result = CuMatrix(A.T, 1, 1)
    n = convert(Int32, A.dims[1] > A.dims[2] ? A.dims[1] : A.dims[2])
    cuda_dot(n, A.ptr, int32(1), B.ptr, int32(1), result.ptr)
    return result
end
