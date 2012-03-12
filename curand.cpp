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

    if (dbl) {
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

    if (dbl) {
        curandGenerateNormal(stream, (float *)ptr, numel, 0, 1);
    } else {
        curandGenerateNormalDouble(stream, (double *)ptr, numel, 0, 1);
    }

    return;
}
