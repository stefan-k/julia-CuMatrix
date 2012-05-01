#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "cuplus.h"


unsigned int cuda_cufftPlan1d(int nx, int type, int batch)
{
    unsigned int ptr;
    cufftPlan1d(&ptr, nx, (cufftType)type, batch);
    return ptr;
}

unsigned int cuda_cufftPlan2d(int nx, int ny, int type)
{
    unsigned int ptr;
    cufftPlan2d(&ptr, nx, ny, (cufftType)type);
    return ptr;
}

unsigned int cuda_cufftPlan3d(int nx, int ny, int nz, int type)
{
    unsigned int ptr;
    cufftPlan3d(&ptr, nx, ny, nz, (cufftType)type);
    return ptr;
}
