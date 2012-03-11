#include <stdio.h>
#define API extern "C" __attribute__((visibility("default")))

API void *cuda_malloc(int bytes);
API void cuda_free(void *ptr);
API void cuda_memcpy_D2H(void *dst, void *src, int bytes);
API void cuda_memcpy_D2D(void *dst, void *src, int bytes);
API void cuda_memcpy_H2D(void *dst, void *src, int bytes);

