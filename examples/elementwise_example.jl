load("cumatrix.jl")
load("elementwise.jl")

x = ElementwiseKernel("test",  "x[i] = y[i]", "float *x, float *y", "")

bla = elementwiseKernelString(x)
kernel_func, libkernel = compile(bla)

print(typeof(kernel_func))

Ah = ones(Float32, 2, 2)
A = CuMatrix(Ah)
B = CuMatrix(Float32, 2,2)

kernel_func(A.ptr, B.ptr)
print(A)
print(B)
