libcuplus=dlopen("libcuplus")
libcublas=dlopen("libcublas")

# Show cuda info
function cuda_info()
    ccall(dlsym(libcuplus, :cuda_info),
          Void, ())
end

# Allocate device memory
function cuda_malloc(T::Type, count::Integer)
    ptr::Ptr{Void} = ccall(dlsym(libcuplus, :cuda_malloc),
                           Ptr{Void}, (Integer, ),
                           count * sizeof(T))
    if ptr == C_NULL
        # If unable to allocate, call garbage collector
        gc()
        # Try allocating again
        ptr::Ptr{Void} = ccall(dlsym(libcuplus, :cuda_malloc),
                               Ptr{Void}, (Integer, ),
                               count * sizeof(T))
    end

    if ptr == C_NULL
        # If still unable to allocate, error out
        error("Can not allocate the required memory")
    end
    return ptr
end
cuda_malloc(T::Type, rows::Integer, cols::Integer) = cuda_malloc(T, rows * cols)
cuda_malloc(T::Type, dims::(Integer, Integer)) = cuda_malloc(T, dims[1] * dims[2])

# Free device memory
function cuda_free(ptr::Ptr{Void})
    ccall(dlsym(libcuplus, :cuda_free),
          Void, (Integer, ),
          ptr)
end

# Copy from Host to Device Memory
function mem_device(dst::Ptr{Void}, src::Matrix)
    bytes::Int32 = sizeof(eltype(src)) * numel(src)
    ccall(dlsym(libcuplus, :cuda_memcpy_h2d),
          Void, (Ptr{Void}, Ptr{Void}, Int32),
          dst, src, bytes)
          
end

# Copy from Device to Host Memory
function mem_host(dst::Matrix, src::Ptr{Void})
    bytes::Int32 = sizeof(eltype(dst)) * numel(dst)
    ccall(dlsym(libcuplus, :cuda_memcpy_d2h),
          Void, (Ptr{Void}, Ptr{Void}, Int32),
          dst, src, bytes)
          
end

# Copy from Device to Device Memory
function mem_copy(dst::Ptr{Void}, src::Ptr{Void}, bytes::Int32)
    ccall(dlsym(libcuplus, :cuda_memcpy_d2d),
          Void, (Ptr{Void}, Ptr{Void}, Int32),
          dst, src, bytes)
end

# Create uniform random Matrix
function cuda_rand(T::Type, ptr::Ptr{Void}, count::Integer)
    count = convert(Int32, count)
    IsFloat64::Bool = (T == Float64)
    ccall(dlsym(libcuplus, :cuda_rand),
          Void, (Ptr{Void}, Int32, Bool),
          ptr, count, IsFloat64)
end

# Create normal random Matrix
function cuda_randn(T::Type, ptr::Ptr{Void}, count::Integer)
    count = convert(Int32, count)
    IsFloat64::Bool = (T == Float64)
    ccall(dlsym(libcuplus, :cuda_randn),
          Void, (Ptr{Void}, Int32, Bool),
          ptr, count, IsFloat64)
end

# Matrix multiply
function cuda_gemm{T}(transA::Char, transB::Char,
                    m::Int32, n::Int32, k::Int32,
                    alpha::T, A::Ptr{Void}, lda::Int32,
                    B::Ptr{Void}, ldb::Int32, beta::T,
                    C::Ptr{Void}, ldc::Int32)
    # FIXME: Fix the following fugliness
    if (T == Float32)
        ccall(dlsym(libcublas, :cublasSgemm),
              Void, (Char, Char, Int32, Int32, Int32,
                     Float32, Ptr{Float32}, Int32, Ptr{Float32}, Int32,
                     Float32, Ptr{Float32}, Int32),
              transA, transB, m, n, k, alpha, A, lda, B, ldb,
              beta, C, ldc)
    else
        ccall(dlsym(libcublas, :cublasDgemm),
              Void, (Char, Char, Int32, Int32, Int32,
                     Float64, Ptr{Float64}, Int32, Ptr{Float64}, Int32,
                     Float64, Ptr{Float64}, Int32),
              transA, transB, m, n, k, alpha, A, lda, B, ldb,
              beta, C, ldc)
    end
end                  

# Find the index of the absolute maximum value
function cuda_amax{T}(num::Int32, x::Ptr{T})
    if (T == Float64)
        ccall(dlsym(libcublas, :cublasIdamax),
              Int32, (Int32, Ptr{Float64}, Int32),
              num, x, 1)
    else
        ccall(dlsym(libcublas, :cublasIsamax),
              Int32, (Int32, Ptr{Float32}, Int32),
              num, x, 1)
    end
end

# Find the index of the absolute minimum value
function cuda_amax{T}(num::Int32, x::Ptr{T})
    if (T == Float64)
        ccall(dlsym(libcublas, :cublasIdamax),
              Int32, (Int32, Ptr{Float64}, Int32),
              num, x, 1)
    else
        ccall(dlsym(libcublas, :cublasIsamax),
              Int32, (Int32, Ptr{Float32}, Int32),
              num, x, 1)
    end
end

# Find the sum of absolute values
function cuda_asum{T}(num::Int32, x::Ptr{Void})
    if (T == Float64)
        ccall(dlsym(libcublas, :cublasDasum),
              Float64, (Int32, Ptr{Float64}, Int32),
              num, x, 1)
    else
        ccall(dlsym(libcublas, :cublasSasum),
              Float32, (Int32, Ptr{Float32}, Int32),
              num, x, 1)
    end
end

# Scale the matrix
function cuda_scal{T}(num::Int32, x::Ptr{Void}, alpha::T)
    if (T == Float64)
        ccall(dlsym(libcublas, :cublasDscal),
              Void, (Int32, Float64, Ptr{Float64}, Int32),
              num, alpha, x, 1)
    else
        ccall(dlsym(libcublas, :cublasSscal),
              Void, (Int32, Float32, Ptr{Float32}, Int32),
              num, alpha, x, 1)
    end
end

# Dot product
function cuda_dot{T}(num::Int32, x::Ptr{T}, y::Ptr{T})
    if (T == Float64)
        ccall(dlsym(libcublas, :cublasDdot),
              Float64, (Int32, Ptr{T}, Int32, Ptr{T}, Int32),
              num, x, 1, y, 1)
    else
        ccall(dlsym(libcublas, :cublasSdot),
              Float32, (Int32, Ptr{T}, Int32, Ptr{T}, Int32),
              num, x, 1, y, 1)
    end 
end
