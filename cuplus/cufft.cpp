#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "cuplus.h"


cufftHandle cuda_cufftPlan1d(int nx, int type, int batch)
{
    cufftHandle ptr;
    cufftPlan1d(&ptr, nx, (cufftType)type, batch);
    return ptr;
}


