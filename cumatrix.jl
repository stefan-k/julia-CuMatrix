include("cuda.jl")

# The CuMatrix class
type CuMatrix
    T::Type
    ptr::Ptr
    dims::(Integer,Integer)

    # Construct Matrix on device
    function CuMatrix(T::Type, dims::(Integer, Integer))
        if T != Float32   && T != Float64 &&
           T != Complex64 && T != Complex128
            error("No integer or boolean support yet")
        end
        ptr = cuda_malloc(T, dims)
        Res = new(T, ptr, dims)
        finalizer(Res, CuFree)
        Res
    end

    # Copy Matrix from host
    function CuMatrix{T}(in::Matrix{T})
        dims = size(in)
        ptr = cuda_malloc(T, dims)
        mem_device(ptr, in)
        Res = new(T, ptr, dims)
        finalizer(Res, CuFree)
        Res
    end

    # Constructor with existing device pointer
    function CuMatrix{T}(ptr::Ptr{T}, dims::(Integer, Integer))
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
    out = Array(eltype(in), in.dims[1], in.dims[2])
    mem_host(out, in.ptr)
    return out
end

# Perform a deep copy
function copy(in::CuMatrix)
    ptr = cuda_malloc(eltype(in), in.dims)
    bytes::Int32 = numel(in) * sizeof(eltype(in))
    mem_copy(ptr, in.ptr, bytes)
    CuMatrix(ptr, in.dims)
end

# Display function
function print(in::CuMatrix)
    print("On GPU\n", Array(in), "\n")
end

# Display function
function show(in::CuMatrix)
    print(in)
end

# Return number of elements
numel(A::CuMatrix) = A.dims[1]*A.dims[2]
numel(T::Type, A::CuMatrix) = convert(T, numel(A))

# Return type of elements
eltype(A::CuMatrix) = A.T

# Freeing memory
function CuFree(in::CuMatrix)
    cuda_free(in.ptr)
end

# Random Number Generators
function curand(T::Type, rows::Integer, cols::Integer)
    out = CuMatrix(T, rows, cols)
    cuda_rand(out.ptr, rows * cols)
    return out
end

function curandn(T::Type, rows::Integer, cols::Integer)
    # curand only supports randn for even number of elements
    count = rows * cols
    count += count & 1 # Increase the malloc size
    ptr = cuda_malloc(T, count)
    cuda_randn(ptr, count)
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
    if (eltype(A) != eltype(B))
        error("Precision mismatch in Matrix multiply")
    end
    C = CuMatrix(eltype(A), (A.dims[1], B.dims[2]))

    m = convert(Int32, C.dims[1])
    n = convert(Int32, C.dims[2])
    k = convert(Int32, B.dims[1])
    
    cuda_gemm('N', 'N', m, n, k,
            one(A.T), A.ptr, m, B.ptr, k,
            zero(A.T), C.ptr, m)
    return C
end

amax(A::CuMatrix) = cuda_amax(numel(Int32, A), A.ptr)
amin(A::CuMatrix) = cuda_amin(numel(Int32, A), A.ptr)
asum(A::CuMatrix) = cuda_asum(numel(Int32, A), A.ptr)

function (*)(A::CuMatrix, alpha)
    n = numel(Int32, A)
    B = copy(A)
    alpha = convert(eltype(A), alpha)
    cuda_scal(n, B.ptr, alpha)
    return B
end

function dot(A::CuMatrix, B::CuMatrix)
    if eltype(A) != eltype(B)
        error("Precision mismatch in Dot product")
    end
   
    n = numel(Int32, A)
    m = numel(Int32, A)

    if m != n
        error("Size mismatch in Dot product")
    end

    cuda_dot(n, A.ptr, B.ptr)
end

# Euclidean norm
nrm2(A::CuMatrix) = cuda_nrm2(numel(Int32, A), A.ptr)

# Thrust functions


# lapack functions
function norm(A::CuMatrix, p)
    if (p == 2)
        nrm2(A::CuMatrix)
    else
        error("norm not supported for p == ", p)
    end
end
