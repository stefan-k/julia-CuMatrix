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
for (fname, elty) in ((:cublasSgemm, :Float32), 
                      (:cublasDgemm, :Float64),
                      (:cublasCgemm, :Complex64), 
                      (:cublasZgemm, :Complex128))
    @eval begin
        function cuda_gemm(transA::Char, transB::Char,
                           m::Int32, n::Int32, k::Int32,
                           alpha::($elty), A::Ptr{$elty}, lda::Int32,
                           B::Ptr{$elty}, ldb::Int32, beta::($elty),
                           C::Ptr{$elty}, ldc::Int32)
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Char, Char, Int32, Int32, Int32,
                         $elty, Ptr{$elty}, Int32, Ptr{$elty}, Int32,
                         $elty, Ptr{$elty}, Int32),
                  transA, transB, m, n, k, alpha, A, lda, B, ldb,
                  beta, C, ldc)
        end
    end
end

# Find the index of the absolute maximum value
for (fname, elty) in ((:cublasIdamax, :Float64), 
                      (:cublasIsamax, :Float32),
                      (:cublasIcamax, :Complex64), 
                      (:cublasIzamax, :Complex128))
    @eval begin
        function cuda_amax(num::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Int32, (Int32, Ptr{$elty}, Int32),
                  num, x, 1)
        end
    end
end

# Find the index of the absolute minimum value
for (fname, elty) in ((:cublasIdamin, :Float64), 
                      (:cublasIsamin, :Float32),
                      (:cublasIcamin, :Complex64), 
                      (:cublasIzamin, :Complex128))
    @eval begin
        function cuda_amin(num::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Int32, (Int32, Ptr{$elty}, Int32),
                  num, x, 1)
        end
    end
end

# Find the sum of absolute values
for (fname, elty) in ((:cublasDasum, :Float64), 
                      (:cublasSasum, :Float32),
                      (:cublasScasum, :Complex64), 
                      (:cublasDzasum, :Complex128))
    @eval begin
        function cuda_asum(num::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  $elty, (Int32, Ptr{$elty}, Int32),
                  num, x, 1)
        end
    end
end

# Scale the matrix
for (fname, elty1, elty2) in ((:cublasSscal, :Float32, :Float32), 
                              (:cublasDscal, :Float64, :Float64),
                              (:cublasCscal, :Complex64, :Complex64),
                              (:cublasCsscal, :Float32, :Complex64),
                              (:cublasZscal, :Complex128, :Complex128),
                              (:cublasZdscal, :Float64, :Complex128))
    @eval begin
        function cuda_scal(num::Int32, x::Ptr{$elty2}, alpha::($elty1))
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, $elty1, Ptr{$elty2}, Int32),
                  num, alpha, x, 1)
        end
    end
end

# Dot product
for (fname, elty) in ((:cublasSdot, :Float32),
                      (:cublasDdot, :Float64),
                      (:cublasCdotu, :Complex64),
                      (:cublasCdotc, :Complex64),
                      (:cublasZdotu, :Complex128),
                      (:cublasZdotc, :Complex128))
    @eval begin
        function cuda_dot(num::Int32, x::Ptr{$elty}, y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  ($elty) , (Int32, Ptr{$elty}, Int32, Ptr{$elty}, Int32),
                  num, x, 1, y, 1)
        end
    end
end

for (fname, elty) in ((:cublasSaxpy, :Float32),
                      (:cublasDaxpy, :Float64),
                      (:cublasCaxpy, :Complex64),
                      (:cublasZaxpy, :Complex128))
    @eval begin
        function cuda_axpy(n::Int32, alpha::($elty), x::Ptr{$elty}, y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, $elty, Ptr{$elty}, Int32, 
                         Ptr{$elty}, Int32),
                  n, alpha, x, 1, y, 1)
        end
    end
end

# copy matrix
for (fname, elty) in ((:cublasScopy, :Float32),
                      (:cublasDcopy, :Float64),
                      (:cublasCcopy, :Complex64),
                      (:cublasZcopy, :Complex128))
    @eval begin
        function cuda_copy(n::Int32, x::Ptr{$elty}, y::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  Void, (Int32, Ptr{$elty}, Int32, 
                         Ptr{$elty}, Int32),
                  n, x, 1, y, 1)
        end
    end
end

# Norm
for (fname, elty) in ((:cublasSnrm2, :Float32),
                      (:cublasDnrm2, :Float64),
                      (:cublasScnrm2, :Complex64),
                      (:cublasDznrm2, :Complex128))
    @eval begin
        function cuda_nrm2(n::Int32, x::Ptr{$elty})
            ccall(dlsym(libcublas, $string(fname)),
                  ($elty), (Int32, Ptr{$elty}, Int32),
                  n, x, 1)
        end
    end
end
