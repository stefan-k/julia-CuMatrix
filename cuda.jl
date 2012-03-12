libCUMEM=dlopen("libcuplus")

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
