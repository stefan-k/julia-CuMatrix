# A lot of this stuff is stolen from PyCUDA
libcuda = dlopen("libcuda")

type ElementwiseKernel
    name::String
    operation::String
    arguments::String
    preamble::String
end

type CuBlock
    x::Integer
    y::Integer
    z::Integer
end

type CuGrid
    x::Integer
    y::Integer
    z::Integer
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

    #system("nvcc -Xcompiler -fPIC -o libkernel.so -shared kernel.cu")
    #libkernel = dlopen("libkernel")

    #return ((a::Ptr{Float32}, b::Ptr{Float32}) -> ccall(dlsym(libkernel, :test), Void, (Ptr{Float32}, Ptr{Float32}), a, b), libkernel)

    system("nvcc --cubin kernel.cu")

    mod = module_from_file("kernel.cubin")

    func = get_function(mod, "test")

    b = CuBlock(256, 1, 1)
    g = CuGrid(2, 1, 1)

    launch_kernel(func, b, g)
end

function module_from_file(file::String)
    ccall(dlsym(libcuplus, :module_from_file),
          Ptr{Void}, (Ptr{Uint8},), cstring(file))
end

function get_function(mod::Ptr{Void}, name::String)
    ccall(dlsym(libcuplus, :get_function),
          Ptr{Void}, (Ptr{Void}, Ptr{Uint8}), mod, cstring(name))
end

function launch_kernel(func::Ptr{Void}, block::CuBlock, grid::CuGrid)
    ccall(dlsym(libcuda, :cuLaunchKernel),
          Void, (Ptr{Void}, Int32, Int32, Int32, Int32, Int32, Int32, Uint32, Int32, Int32, Ptr{Void}),
          func, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, 0, 0, C_NULL)
end
