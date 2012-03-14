libCUMEM=dlopen("libcuplus")
libCUBLAS=dlopen("libcublas")

# Allocate device memory
function jl_cuda_malloc(T::Type, count::Integer)
    ptr::Ptr{Void} = ccall(dlsym(libCUMEM, :cuda_malloc),
                           Ptr{Void}, (Integer, ),
                           count * sizeof(T))
    return ptr
end
jl_cuda_malloc(T::Type, rows::Integer, cols::Integer) = jl_cuda_malloc(T, rows * cols)
jl_cuda_malloc(T::Type, dims::(Integer, Integer)) = jl_cuda_malloc(T, dims[1] * dims[2])


# Free device memory
function jl_cuda_free(ptr::Ptr{Void})
    ccall(dlsym(libCUMEM, :cuda_free),
          Void, (Integer, ),
          ptr)
end

# Copy from Host to Device Memory
function jl_mem_device(dst::Ptr{Void}, src::Matrix)
    bytes::Int32 = sizeof(eltype(src)) * numel(src)
    ccall(dlsym(libCUMEM, :cuda_memcpy_h2d),
          Void, (Ptr{Void}, Ptr{Void}, Int32),
          dst, src, bytes)
          
end

# Copy from Device to Host Memory
function jl_mem_host(dst::Matrix, src::Ptr{Void})
    bytes::Int32 = sizeof(eltype(dst)) * numel(dst)
    ccall(dlsym(libCUMEM, :cuda_memcpy_d2h),
          Void, (Ptr{Void}, Ptr{Void}, Int32),
          dst, src, bytes)
          
end

# Copy from Device to Device Memory
function jl_mem_copy(dst::Ptr{Void}, src::Ptr{Void}, bytes::Int32)
    ccall(dlsym(libCUMEM, :cuda_memcpy_d2d),
          Void, (Ptr{Void}, Ptr{Void}, Int32),
          dst, src, bytes)
end

# Create uniform random Matrix
function jl_curand(T::Type, ptr::Ptr{Void}, count::Integer)
    count = convert(Int32, count)
    IsFloat64::Bool = (T == Float64)
    ccall(dlsym(libCUMEM, :cuda_rand),
          Void, (Ptr{Void}, Int32, Bool),
          ptr, count, IsFloat64)
end

# Create normal random Matrix
function jl_curandn(T::Type, ptr::Ptr{Void}, count::Integer)
    count = convert(Int32, count)
    IsFloat64::Bool = (T == Float64)
    ccall(dlsym(libCUMEM, :cuda_randn),
          Void, (Ptr{Void}, Int32, Bool),
          ptr, count, IsFloat64)
end

function jl_gemm{T}(transA::Char, transB::Char,
                    m::Int32, n::Int32, k::Int32,
                    alpha::T, A::Ptr{Void}, lda::Int32,
                    B::Ptr{Void}, ldb::Int32, beta::T,
                    C::Ptr{Void}, ldc::Int32)
    # FIXME: Fix the following fugliness
    if (T == Float32)
        ccall(dlsym(libCUBLAS, :cublasSgemm),
              Void, (Char, Char, Int32, Int32, Int32,
                     Float32, Ptr{Float32}, Int32, Ptr{Float32}, Int32,
                     Float32, Ptr{Float32}, Int32),
              transA, transB, m, n, k, alpha, A, lda, B, ldb,
              beta, C, ldc)
    else
        ccall(dlsym(libCUBLAS, :cublasDgemm),
              Void, (Char, Char, Int32, Int32, Int32,
                     Float64, Ptr{Float64}, Int32, Ptr{Float64}, Int32,
                     Float64, Ptr{Float64}, Int32),
              transA, transB, m, n, k, alpha, A, lda, B, ldb,
              beta, C, ldc)
    end
end                  

function jl_amax{T}(n::Int32, x::Ptr{Void}, incx::Int32, result::Ptr{Void})
    if T == Float64
        ccall(dlsym(libCUBLAS, :cublasIdamax),
              Void, (Int32, Ptr{Float64}, Int32, Ptr{Int32}),
              n, x, incx, result)
    end
    return result
end

function jl_asum{T}(n::Int32, x::Ptr{Void}, incx::Int32, result::Ptr{Void})
    if T == Float64
        ccall(dlsym(libCUBLAS, :cublasDasum),
              Void, (Int32, Ptr{Float64}, Int32, Ptr{Int32}),
              n, x, incx, result)
    end
    return result
end

function jl_scal{T}(n::Int32, alpha::T, x::Ptr{Void}, incx::Int32)
    if T == Float64
        ccall(dlsym(libCUBLAS, :cublasDscal),
              Void, (Int32, Float64, Ptr{Float64}, Int32),
              n, alpha, x, incx)
    end
end
