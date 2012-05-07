#include <stdio.h>
#include <cublas.h>
#define API extern "C" __attribute__((visibility("default")))

// CUDART+ functions
API void *cuda_malloc(int bytes);
API void cuda_free(void *ptr);
API void cuda_memcpy_d2h(void *dst, void *src, int bytes);
API void cuda_memcpy_d2d(void *dst, void *src, int bytes);
API void cuda_memcpy_h2d(void *dst, void *src, int bytes);

// CURAND+ functions
API void cudaSrand(float   *ptr, int numel);
API void cudaDrand(cuComplex  *ptr, int numel);
API void cudaCrand(double  *ptr, int numel);
API void cudaZrand(cuDoubleComplex *ptr, int numel);
API void cudaSrandn(float   *ptr, int numel);
API void cudaCrandn(cuComplex  *ptr, int numel);
API void cudaDrandn(double  *ptr, int numel);
API void cudaZrandn(cuDoubleComplex *ptr, int numel);

// CUFFT functions
API unsigned int cuda_cufftPlan1d(int nx, int type, int batch);
API unsigned int cuda_cufftPlan2d(int nx, int ny, int type);
API unsigned int cuda_cufftPlan3d(int nx, int ny, int nz, int type);
API unsigned int cuda_cufftPlanMany(int rank, int *n, int *inembed, int istride,
                                    int idist, int *onembed, int ostride,
                                    int odist, int type, int batch);
API int cuda_cufftGetVersion();

// Return error
API int cuda_last_error();

// Other
API void cuda_info();
