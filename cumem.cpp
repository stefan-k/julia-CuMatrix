#include <stdio.h>
#include <cuda_runtime_api.h>
#include "cumem.h"

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

#define CUDAMEMCPY(TYPE, CUDATYPE)                              \
    void cuda_memcpy_##TYPE(void *dst, void *src, int bytes)    \
    {                                                           \
        cudaMemcpy(dst, src, bytes, CUDATYPE);                  \
        return;                                                 \
    }                                                           \

CUDAMEMCPY(D2H, cudaMemcpyDeviceToHost)
CUDAMEMCPY(H2D, cudaMemcpyHostToDevice)
CUDAMEMCPY(D2D, cudaMemcpyDeviceToDevice)
