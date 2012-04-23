#include <stdio.h>
#include <cuda_runtime_api.h>
#include "cuplus.h"

int cuda_last_error()
{
    return (int)cudaGetLastError();
}

void cuda_info()
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }

    if (count == 0) {
        printf("No CUDA capable devices found\n");
        return;
    }

    int drv_ver = 0, rt_ver = 0;
    cudaDriverGetVersion(&drv_ver);
    cudaRuntimeGetVersion(&rt_ver);

    if(drv_ver < rt_ver) {
        printf("Driver and runtime versions incompatible\n");
        return;
    }

    printf("CUDA version: %d.%d\n", rt_ver/1000, (rt_ver%100)/10);

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: Name - %s. Compute - %d.%d. Memory - %d MB.\n",
               i, prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 20);
    }
}
