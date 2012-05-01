#include <stdio.h>
#define API extern "C" __attribute__((visibility("default")))

// CUDART+ functions
API void *cuda_malloc(int bytes);
API void cuda_free(void *ptr);
API void cuda_memcpy_d2h(void *dst, void *src, int bytes);
API void cuda_memcpy_d2d(void *dst, void *src, int bytes);
API void cuda_memcpy_h2d(void *dst, void *src, int bytes);

// CURAND+ functions
API void cuda_rand (void *ptr, int numel, bool dbl);
API void cuda_randn(void *ptr, int numel, bool dbl);

// CUFFT functions
API unsigned int cuda_cufftPlan1d(int nx, int type, int batch);
API unsigned int cuda_cufftPlan2d(int nx, int ny, int type);
API unsigned int cuda_cufftPlan3d(int nx, int ny, int nz, int type);

// Return error
API int cuda_last_error();

// Other
API void cuda_info();
