libcuplus=dlopen("libcuplus")

# Show cuda info
function cuda_info()
    ccall(dlsym(libcuplus, :cuda_info),
          Void, ())
end

# Allocate device memory
function cuda_malloc{T}(T::Type{T}, count::Integer)
    ptr = ccall(dlsym(libcuplus, :cuda_malloc),
                Ptr{T}, (Int32, ),
                count * sizeof(T))
    if ptr == C_NULL
        # If unable to allocate, call garbage collector
        gc()
        # Try allocating again
        ptr = ccall(dlsym(libcuplus, :cuda_malloc),
                    Ptr{T}, (Int32, ),
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
function cuda_free{T}(ptr::Ptr{T})
    ccall(dlsym(libcuplus, :cuda_free),
          Void, (Ptr{T}, ),
          ptr)
end

# Copy from Host to Device Memory
function mem_device{T}(dst::Ptr{T}, src::Matrix{T})
    bytes::Int32 = sizeof(T) * numel(src)
    ccall(dlsym(libcuplus, :cuda_memcpy_h2d),
          Void, (Ptr{T}, Ptr{T}, Int32),
          dst, src, bytes)
end

# Copy from Device to Host Memory
function mem_host{T}(dst::Matrix, src::Ptr{T})
    bytes::Int32 = sizeof(T) * numel(dst)
    ccall(dlsym(libcuplus, :cuda_memcpy_d2h),
          Void, (Ptr{T}, Ptr{T}, Int32),
          dst, src, bytes)
end

# Copy from Device to Device Memory
function mem_copy{T}(dst::Ptr{T}, src::Ptr{T}, bytes::Int32)
    ccall(dlsym(libcuplus, :cuda_memcpy_d2d),
          Void, (Ptr{T}, Ptr{T}, Int32),
          dst, src, bytes)
end

# Create uniform random Matrix
for (fname, elty) in ((:cudaSrand, :Float32),
                      (:cudaDrand, :Float64),
                      (:cudaCrand, :Complex64),
                      (:cudaZrand, :Complex128))
    @eval begin
        function cuda_rand(ptr::Ptr{$elty}, count::Integer)
            count = convert(Int32, count)
            ccall(dlsym(libcuplus, $string(fname)),
                  Void, (Ptr{$elty}, Int32),
                  ptr, count)
        end
    end
end

# Create normal random Matrix
for (fname, elty) in ((:cudaSrandn, :Float32),
                      (:cudaDrandn, :Float64),
                      (:cudaCrandn, :Complex64),
                      (:cudaZrandn, :Complex128))
    @eval begin
        function cuda_randn(ptr::Ptr{$elty}, count::Integer)
            count = convert(Int32, count)
            ccall(dlsym(libcuplus, $string(fname)),
                  Void, (Ptr{$elty}, Int32),
                  ptr, count)
        end
    end
end
