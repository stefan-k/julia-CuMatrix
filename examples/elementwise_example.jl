load("cumatrix.jl")
load("elementwise.jl")

x = ElementwiseKernel("test",  "x[i] = y[i]", "float *x, float *y", "")

bla = elementwiseKernelString(x)
print(compile(bla))

Ah = ones(Float32, 2, 2)
A = CuMatrix(Ah)
B = CuMatrix(Float32, 2,2)

#kernel_func(A.ptr, B.ptr)
#print(A)
#print(B)
