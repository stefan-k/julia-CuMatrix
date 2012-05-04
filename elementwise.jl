# A lot of this stuff is stolen from PyCUDA

type ElementwiseKernel
    name::String
    operation::String
    arguments::String
    preamble::String
end

function elementwiseKernelString(k::ElementwiseKernel)
    # multiline strings would be nice ;)
    kernel = "$(k.preamble)\n\n__global__ void $(k.name)($(k.arguments))\n{\n  unsigned tid = threadIdx.x;\n  unsigned total_threads = gridDim.x*blockDim.x;\n  unsigned cta_start = blockDim.x*blockIdx.x; \n  unsigned i;\n\n  for(i = cta_start + tid; i < 1/*n*/; i += total_threads)\n  {\n    $(k.operation);\n  }\n}"
    return kernel
end

function compile(source::String)
    source = "extern \"C\" {\n$(source)\n}\n"
    file = open("kernel.cu", "w")
    write(file, source)
    close(file)

    system("nvcc -Xcompiler -fPIC -o libkernel.so -shared kernel.cu")
    libkernel = dlopen("libkernel")

    return ((a::Ptr{Float32}, b::Ptr{Float32}) -> ccall(dlsym(libkernel, :test), Void, (Ptr{Float32}, Ptr{Float32}), a, b), libkernel)
end
