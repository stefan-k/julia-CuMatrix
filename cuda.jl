libCUMEM=dlopen("libcumem")

function jl_cuda_malloc(T::Type, m::Integer, n::Integer)
    ptr::Ptr{Void} = ccall(dlsym(libCUMEM, :cuda_malloc),
                           Ptr{Void}, (Integer, ),
                           m * n * sizeof(T))
    print(ptr,"\n")
    return ptr
end

function jl_cuda_free(ptr::Ptr{Void})
    ccall(dlsym(libCUMEM, :cuda_free),
          Void, (Integer, ),
          ptr)
end
