#include <stdio.h>
//#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuplus.h"

void *module_from_file(const char* filename)
{
  CUmodule *mod;
  //CUDAPP_CALL_GUARDED(cuModuleLoad, (&mod, filename));
  cuModuleLoad(mod, filename);
  return mod;
}

void *get_function(CUmodule module, const char* name)
{
  CUfunction *func;
  cuModuleGetFunction(func, module, name);
  return func;
}
