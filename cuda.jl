libCUMEM=dlopen("libcuplus")

function jl_cuda_malloc(T::Type, m::Integer, n::Integer)
    ptr::Ptr{Void} = ccall(dlsym(libCUMEM, :cuda_malloc),
                           Ptr{Void}, (Integer, ),
                           m * n * sizeof(T))
    return ptr
end

function jl_cuda_free(ptr::Ptr{Void})
    ccall(dlsym(libCUMEM, :cuda_free),
          Void, (Integer, ),
          ptr)
end

function jl_mem_device(dst::Ptr{Void}, src::Matrix)
    bytes::Int32 = sizeof(eltype(src)) * numel(src)
    ccall(dlsym(libCUMEM, :cuda_memcpy_h2d),
          Void, (Ptr{Void}, Ptr{Void}, Int32),
          dst, src, bytes)
          
end

function jl_mem_host(dst::Matrix, src::Ptr{Void})
    bytes::Int32 = sizeof(eltype(dst)) * numel(dst)
    ccall(dlsym(libCUMEM, :cuda_memcpy_d2h),
          Void, (Ptr{Void}, Ptr{Void}, Int32),
          dst, src, bytes)
          
end

function jl_mem_copy(dst::Ptr{Void}, src::Ptr{Void}, bytes::Int32)
    ccall(dlsym(libCUMEM, :cuda_memcpy_d2d),
          Void, (Ptr{Void}, Ptr{Void}, Int32),
          dst, src, bytes)
end

jl_cuda_malloc(T::Type, dims::(Integer, Integer)) = jl_cuda_malloc(T, dims[1], dims[2])
