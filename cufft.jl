libcufft = dlopen("libcufft")
libcuplus = dlopen("libcuplus")

CUFFT_R2C = int32(0x2a)
CUFFT_C2R = int32(0x2c)
CUFFT_C2C = int32(0x29)
CUFFT_D2Z = int32(0x6a)
CUFFT_Z2D = int32(0x6c)
CUFFT_Z2Z = int32(0x69)

function cufftPlan1d(nx::Int32, cufft_type::Int32, batch::Int32)
    ccall(dlsym(libcuplus, :cuda_cufftPlan1d),
          Uint32, (Int32, Int32, Int32),
          nx, cufft_type, batch)
end

function cufftPlan2d(nx::Int32, ny::Int32, cufft_type::Int32)
    ccall(dlsym(libcuplus, :cuda_cufftPlan2d),
          Uint32, (Int32, Int32, Int32),
          nx, ny, cufft_type)
end

function cufftPlan3d(nx::Int32, ny::Int32, nz::Int32, cufft_type::Int32)
    ccall(dlsym(libcuplus, :cuda_cufftPlan3d),
          Uint32, (Int32, Int32, Int32, Int32),
          nx, ny, nz, cufft_type)
end
