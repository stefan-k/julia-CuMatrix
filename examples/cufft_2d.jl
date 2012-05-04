## Example showing usage of the CUFFT wrappers
# Author: Stefan Kroboth

load("cumatrix.jl")

print("CUFFT version: $(cufftGetVersion())\n\n")

sx, sy = 16, 16
#sx, sy = 512, 512
#sx, sy = 1024, 1024

# create CUFFT plan
plan = cufftPlan2d(sx, sy, CUFFT_C2C)

# create data
h_data = ones(Complex64, sx, sy)
d_data = CuMatrix(h_data)  # from host to device
o_data = copy(d_data)
d_data_inv = copy(d_data)

# FFT
print("FFT:\n")
@time cufftExec(plan, d_data, o_data, CUFFT_FORWARD)
print(o_data)
println()

# IFFT
print("IFFT:\n")
@time cufftExec(plan, o_data, d_data_inv, CUFFT_INVERSE)
print(d_data_inv*float32(1.0/(sx*sy)))
