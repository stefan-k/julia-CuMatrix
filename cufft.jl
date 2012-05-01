libcufft = dlopen("libcufft")
libcuplus = dlopen("libcuplus")

#load("cumatrix.jl") # this has to be solved in another way

CUFFT_R2C = 0x2a
CUFFT_C2R = 0x2c
CUFFT_C2C = 0x29
CUFFT_D2Z = 0x6a
CUFFT_Z2D = 0x6c
CUFFT_Z2Z = 0x69

#typealias cufftHandle Uint32
#typealias CufftResult Uint8
#typealias CufftType Uint8

#type CufftHandle
    #ptr::Ptr{Uint32}

    #function CufftHandle()
        #ptr = cuda_malloc(Uint32, 1)
        #new(ptr)
    #end
#end

## Copy from Device to Host Memory
#function mem_host{T}(dst::T, src::Ptr{T})
    #bytes::Int32 = sizeof(T) * numel(dst)
    #ccall(dlsym(libcuplus, :cuda_memcpy_d2h),
          #Void, (Ptr{T}, Ptr{T}, Int32),
          #dst, src, bytes)
#end

#function get(a::CufftHandle)
    #out = uint32(0)
    #mem_host(out, a.ptr)
    #return out
#end

#function cufftPlan1d(plan::CufftHandle, nx::Int32, cufft_type::CufftType, batch::Int32)
    #ccall(dlsym(libcufft, :cufftPlan1d),
          #CufftResult, (Ptr{Uint32}, Int32, CufftType, Int32),
          #&plan.ptr, nx, cufft_type, batch)
#end

function cufftPlan1d(nx::Int32, cufft_type::Int32, batch::Int32)
    ccall(dlsym(libcuplus, :cuda_cufftPlan1d),
          Uint8, (Int32, Int32, Int32),
          nx, int32(cufft_type), batch)
end

#function cufftPlan2d(plan::CufftHandle, nx::Int32, ny::Int32, cufft_type::Uint8, batch::Int32)
    #ccall(dlsym(libcufft, :cufftPlan2d),
          #CufftResult, (Ptr{CufftHandle}, Int32, Int32, CufftType, Int32),
          #plan, nx, ny, cufft_type, batch)
#end
