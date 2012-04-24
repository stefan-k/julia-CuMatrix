#include <stdio.h>
#include <cuda_runtime_api.h>
#include "cuplus.h"

void *cuda_malloc(int bytes)
{
    void *ptr = NULL;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void cuda_free(void *ptr)
{
    cudaFree((void *)ptr);
    return;
}

#define CUDAMEMCPY(TYPE, MEMCPY_TYPE)                           \
    void cuda_memcpy_##TYPE(void *dst, void *src, int bytes)    \
    {                                                           \
        cudaMemcpy(dst, src, bytes, MEMCPY_TYPE);               \
        return;                                                 \
    }                                                           \

CUDAMEMCPY(d2h, cudaMemcpyDeviceToHost)
CUDAMEMCPY(h2d, cudaMemcpyHostToDevice)
CUDAMEMCPY(d2d, cudaMemcpyDeviceToDevice)
