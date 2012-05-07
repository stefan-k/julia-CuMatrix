libcufft = dlopen("libcufft")
libcuplus = dlopen("libcuplus")

CUFFT_R2C = int32(0x2a)
CUFFT_C2R = int32(0x2c)
CUFFT_C2C = int32(0x29)
CUFFT_D2Z = int32(0x6a)
CUFFT_Z2D = int32(0x6c)
CUFFT_Z2Z = int32(0x69)

CUFFT_FORWARD = int32(-1)
CUFFT_INVERSE = int32(1)

CUFFT_COMPATIBILITY_NATIVE          = int32(0x00)
CUFFT_COMPATIBILITY_FFTW_PADDING    = int32(0x01)
CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = int32(0x02)
CUFFT_COMPATIBILITY_FFTW_ALL        = int32(0x03)

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

function cufftPlanMany(rank::Int32, n::Ptr{Int32}, inembed::Ptr{Int32}, istride::Int32, 
                       idist::Int32, onembed::Ptr{Int32}, ostride::Int32, odist::Int32,
                       cufft_type::Int32, batch::Int32)
    ccall(dlsym(libcuplus, :cuda_cufftPlanMany),
          Uint32, (Int32, Ptr{Int32}, Ptr{Int32}, Int32, Int32, Ptr{Int32}, Int32, Int32, Int32, Int32),
          rank, n, inembed, istride, idist, onembed, ostride, odist, cufft_type, batch)
end

cufftPlan1d(nx::Integer, cufft_type::Int32, batch::Integer) = cufftPlan1d(int32(nx), cufft_type, int32(batch))
cufftPlan1d(nx::Integer, cufft_type::Int32) = cufftPlan1d(int32(nx), cufft_type, int32(1))
cufftPlan1d(nx::Integer) = cufftPlan1d(int32(nx), CUFFT_C2C, int32(1))
cufftPlan2d(nx::Integer, ny::Integer, cufft_type::Int32) = cufftPlan2d(int32(nx), int32(ny), cufft_type)
cufftPlan2d(nx::Integer, ny::Integer) = cufftPlan2d(int32(nx), int32(ny), CUFFT_C2C)
cufftPlan3d(nx::Integer, ny::Integer, nz::Integer, cufft_type::Int32) = cufftPlan2d(int32(nx), int32(ny), int(nz), cufft_type)
cufftPlan3d(nx::Integer, ny::Integer, nz::Integer) = cufftPlan2d(int32(nx), int32(ny), int(nz), CUFFT_C2C)

function cufftDestroy(plan::Uint32)
    ccall(dlsym(libcufft, :cufftDestroy),
          Void, (Uint32,), plan)
end

function cufftExecC2C(plan::Uint32, idata::Ptr{Complex64}, odata::Ptr{Complex64}, direction::Int32)
    ccall(dlsym(libcufft, :cufftExecC2C),
          Void, (Uint32, Ptr{Complex64}, Ptr{Complex64}, Int32),
          plan, idata, odata, direction)
end

function cufftExecR2C(plan::Uint32, idata::Ptr{Float32}, odata::Ptr{Complex64})
    ccall(dlsym(libcufft, :cufftExecR2C),
          Void, (Uint32, Ptr{Float32}, Ptr{Complex64}),
          plan, idata, odata)
end

function cufftExecC2R(plan::Uint32, idata::Ptr{Complex64}, odata::Ptr{Float32})
    ccall(dlsym(libcufft, :cufftExecC2R),
          Void, (Uint32, Ptr{Complex64}, Ptr{Float32}),
          plan, idata, odata)
end

function cufftExecZ2Z(plan::Uint32, idata::Ptr{Complex128}, odata::Ptr{Complex128}, direction::Int32)
    ccall(dlsym(libcufft, :cufftExecZ2Z),
          Void, (Uint32, Ptr{Complex128}, Ptr{Complex128}, Int32),
          plan, idata, odata, direction)
end

function cufftExecD2Z(plan::Uint32, idata::Ptr{Float64}, odata::Ptr{Complex128})
    ccall(dlsym(libcufft, :cufftExecC2Z),
          Void, (Uint32, Ptr{Float64}, Ptr{Complex128}),
          plan, idata, odata)
end

function cufftExecZ2D(plan::Uint32, idata::Ptr{Complex128}, odata::Ptr{Float64})
    ccall(dlsym(libcufft, :cufftExecZ2D),
          Void, (Uint32, Ptr{Complex128}, Ptr{Float64}),
          plan, idata, odata)
end

function cufftSetCompatibilityMode(plan::Uint32, mode::Int32)
    ccall(dlsym(libcufft, :cufftSetCompatibilityMode),
          Void, (Uint32, Int32), plan, mode)
end

function cufftGetVersion()
    ccall(dlsym(libcuplus, :cuda_cufftGetVersion),
          Int32, ())
end
