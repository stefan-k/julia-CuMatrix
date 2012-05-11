load("cuda.jl")
load("cufft.jl")

#abstract AbstractCuArray{T,N}
abstract AbstractCuArray

typealias CuLimit Union(Float,Complex)

# The CuArray class
type CuArray{T<:CuLimit,N} <: AbstractCuArray
    ptr::Ptr{T}
    dims::NTuple{N,Int}

    function CuArray(data::Array, dims)
        ptr = cuda_malloc(T, dims)
        mem_device(ptr, data)
        Res = new(ptr, dims)
        finalizer(Res, CuFree)
        Res
    end

    # Construct Matrix on device
    function CuArray(dims...)
        ptr = cuda_malloc(T, dims)
        Res = new(ptr, dims)
        finalizer(Res, CuFree)
        Res
    end

    # Constructor with existing device pointer
    function CuArray{T}(ptr::Ptr{T}, dims...)
        Res = new(ptr, dims...)
        finalizer(Res, CuFree)
        Res
    end

    ## Default type: Single precision (for compatibility with older cards)
    #CuArray(rows::Integer, cols::Integer) = CuArray{Float32}(rows, cols)
    #CuArray(rows::Integer) = CuArray(Float32, rows, 1)
end

function CuArray{T,N}(in::Array{T,N})
    dims = size(in)
    CuArray{T,N}(in, dims)
end

typealias CuMatrix{T} CuArray{T,2} 
typealias CuVector{T} CuArray{T,1} 

# Get matrix from device to host
function Array{T}(in::CuArray{T})
    out = Array(T, in.dims)
    mem_host(out, in.ptr)
    return out
end

# Perform a deep copy
function copy{T}(a::CuArray{T})
    ptr = cuda_malloc(T, a.dims)
    bytes::Int32 = numel(a) * sizeof(eltype(a))
    mem_copy(ptr, a.ptr, bytes)
    CuArray{T,length(a.dims)}(ptr, a.dims)
end

# Display function
function print(in::CuArray)
    print("On GPU\n", Array(in), "\n")
end

# Display function
function show(in::CuArray)
    print(in)
end

# Return number of elements
numel(A::CuArray) = prod(A.dims)
numel(T::Type, A::CuArray) = convert(T, numel(A))

# Return type of elements
eltype{T}(::CuArray{T}) = T

# Freeing memory
function CuFree(in::CuArray)
    cuda_free(in.ptr)
end

# Random Number Generators
#function curand(T::Type, rows::Integer, cols::Integer)
function curand(T::Type, dims...)
    out = CuArray{T}(dims...)
    cuda_rand(out.ptr, prod(dims))
    return out
end

function curandn(T::Type, dims...)
    # curand only supports randn for even number of elements
    count = prod(dims)
    count += count & 1 # Increase the malloc size
    ptr = cuda_malloc(T, count)
    cuda_randn(ptr, count)
    # But only use the required number of elements
    CuArray{T,length(dims)}(ptr, dims)
end

curand(dims...) = curand(Float32, dims)
curandn(dims...) = curandn(Float32, dims)

# BLAS Functions
function (*){T<:CuLimit}(A::CuMatrix{T}, B::CuMatrix{T})
    if (A.dims[2] != B.dims[1]) 
        error("Inner dimension mismatch in Matrix multiply")
    end

    C = CuArray{eltype(A),2}(A.dims[1], B.dims[2])

    m = convert(Int32, C.dims[1])
    n = convert(Int32, C.dims[2])
    k = convert(Int32, B.dims[1])
    
    cuda_gemm('N', 'N', m, n, k, one(T), A.ptr, m, B.ptr, k, zero(T), C.ptr, m)
    return C
end

amax(A::CuArray) = cuda_amax(numel(Int32, A), A.ptr)
amin(A::CuArray) = cuda_amin(numel(Int32, A), A.ptr)
asum(A::CuArray) = cuda_asum(numel(Int32, A), A.ptr)

function (*){T<:CuLimit}(A::CuArray{T}, alpha::T)
    n = numel(Int32, A)
    B = copy(A)
    cuda_scal(n, B.ptr, alpha)
    return B
end

(*){T<:CuLimit}(alpha::T, A::CuArray{T}) = (*)(A::CuArray, alpha)

function dot{T}(A::CuArray{T}, B::CuArray{T})
    n = numel(Int32, A)
    m = numel(Int32, A)

    if m != n
        error("Size mismatch in Dot product")
    end

    cuda_dot(n, A.ptr, B.ptr)
end

# Euclidean norm
nrm2(A::CuArray) = cuda_nrm2(numel(Int32, A), A.ptr)

# Thrust functions


# lapack functions
function norm(A::CuArray, p::Integer)
    if (p == 2)
        nrm2(A::CuArray)
    else
        error("norm not supported for p == ", p)
    end
end

norm(A::CuArray) = norm(A::CuArray, 2)

# CUFFT functions
cufftExec(plan::Uint32, idata::Ptr{Complex64}, odata::Ptr{Complex64}, direction::Int32) = cufftExecC2C(plan, idata, odata, direction)
cufftExec(plan::Uint32, idata::Ptr{Float32}, odata::Ptr{Complex64}) = cufftExecR2C(plan, idata, odata)
cufftExec(plan::Uint32, idata::Ptr{Complex64}, odata::Ptr{Float32}) = cufftExecC2R(plan, idata, odata)
cufftExec(plan::Uint32, idata::Ptr{Complex128}, odata::Ptr{Complex128}, direction::Int32) = cufftExecZ2Z(plan, idata, odata, direction)
cufftExec(plan::Uint32, idata::Ptr{Float64}, odata::Ptr{Complex128}) = cufftExecD2Z(plan, idata, odata)
cufftExec(plan::Uint32, idata::Ptr{Complex128}, odata::Ptr{Float64}) = cufftExecZ2D(plan, idata, odata)
cufftExec(plan::Uint32, idata::CuArray, odata::CuArray, direction::Int32) = cufftExec(plan, idata.ptr, odata.ptr, direction)
cufftExec(plan::Uint32, idata::CuArray, odata::CuArray) = cufftExec(plan, idata.ptr, odata.ptr)
