#include <stdio.h>
#include <curand.h>
#include "cuplus.h"

void cuda_rand(void *ptr, int numel, bool dbl)
{
    static bool is_init = false;
    static curandGenerator_t stream;
    if (!is_init) {
        curandCreateGenerator(&stream, CURAND_RNG_PSEUDO_DEFAULT);
        is_init = true;
    }

    if (!dbl) {
        curandGenerateUniform(stream, (float *)ptr, numel);
    } else {
        curandGenerateUniformDouble(stream, (double *)ptr, numel);
    }

    return;
}


void cuda_randn(void *ptr, int numel, bool dbl)
{
    static bool is_init = false;
    static curandGenerator_t stream;
    if (!is_init) {
        curandCreateGenerator(&stream, CURAND_RNG_PSEUDO_DEFAULT);
        is_init = true;
    }

    if (!dbl) {
        curandGenerateNormal(stream, (float *)ptr, numel, 0, 1);
    } else {
        curandGenerateNormalDouble(stream, (double *)ptr, numel, 0, 1);
    }

    return;
}

void cudaSrand(float *ptr, int numel)
{
    return cuda_rand((void *)ptr, numel, false);
}

void cudaDrand(double *ptr, int numel)
{
    return cuda_rand((void *)ptr, numel, true);
}

void cudaCrand(cuComplex *ptr, int numel)
{
    return cuda_rand((void *)ptr, 2*numel, false);
}

void cudaZrand(cuDoubleComplex *ptr, int numel)
{
    return cuda_rand((void *)ptr, 2*numel, true);
}

void cudaSrandn(float *ptr, int numel)
{
    return cuda_randn((void *)ptr, numel, false);
}

void cudaDrandn(double *ptr, int numel)
{
    return cuda_randn((void *)ptr, numel, true);
}

void cudaCrandn(cuComplex *ptr, int numel)
{
    return cuda_randn((void *)ptr, 2*numel, false);
}

void cudaZrandn(cuDoubleComplex *ptr, int numel)
{
    return cuda_randn((void *)ptr, 2*numel, true);
}
