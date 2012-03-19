include("cumatrix.jl")

# Random Number Generators
function curand(T::Type, rows::Integer, cols::Integer)
    out = CuMatrix(T, rows, cols)
    jl_curand(T, out.ptr, rows * cols)
    return out
end

function curandn(T::Type, rows::Integer, cols::Integer)
    # curand only supports randn for even number of elements
    count = rows * cols
    count += count & 1 # Increase the malloc size
    ptr = jl_cuda_malloc(T, count)
    jl_curandn(T, ptr, count)
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
    
    jl_gemm('N', 'N', m, n, k,
            one(A.T), A.ptr, m, B.ptr, k,
            zero(A.T), C.ptr, m)
    return C
end

function cuamax(A::CuMatrix)
    result = CuMatrix(Int32, 1, 1)
    # needs some more checking
    n = convert(Int32, A.dims[1] > A.dims[2] ? A.dims[1] : A.dims[2])
    jl_amax(n,  A.ptr, int32(1), result.ptr)
    return result
end

function cuasum(A::CuMatrix)
    result = CuMatrix(A.T, 1, 1)
    # needs some more checking
    n = convert(Int32, A.dims[1] > A.dims[2] ? A.dims[1] : A.dims[2])
    jl_asum(n,  A.ptr, int32(1), result.ptr)
    return result
end

function cuscal(A::CuMatrix, alpha)
    # needs some more checking
    n = convert(Int32, A.dims[1] > A.dims[2] ? A.dims[1] : A.dims[2])
    jl_scal(n, alpha, A.ptr, int32(1))
end

function cudot(A::CuMatrix, B::CuMatrix)
    result = CuMatrix(A.T, 1, 1)
    n = convert(Int32, A.dims[1] > A.dims[2] ? A.dims[1] : A.dims[2])
    jl_dot(n, A.ptr, int32(1), B.ptr, int32(1), result.ptr)
    return result
end
