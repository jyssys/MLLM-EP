/* CPU ABI regression. No CUDA runtime or driver call. */
#include <stddef.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

_Static_assert(CUDART_VERSION == 12090, "runtime header version");
typedef cudaError_t (*get_params_signature)(cudaGraphNode_t, struct cudaKernelNodeParams *);
typedef cudaError_t (*get_info_signature)(const void *, size_t, size_t *, size_t *);
typedef cudaError_t (*get_name_signature)(const char **, const void *);
_Static_assert(_Generic(&cudaGraphKernelNodeGetParams, get_params_signature: 1, default: 0), "getter signature");
_Static_assert(_Generic(&cudaFuncGetParamInfo, get_info_signature: 1, default: 0), "parameter-info signature");
_Static_assert(_Generic(&cudaFuncGetName, get_name_signature: 1, default: 0), "name signature");

_Static_assert(sizeof(void *) == 8, "pointer size");
_Static_assert(sizeof(struct cudaKernelNodeParams) == 56, "kernel-node size");
_Static_assert(_Alignof(struct cudaKernelNodeParams) == 8, "kernel-node alignment");
_Static_assert(offsetof(struct cudaKernelNodeParams, func) == 0, "function offset");
_Static_assert(offsetof(struct cudaKernelNodeParams, gridDim) == 8, "grid offset");
_Static_assert(offsetof(struct cudaKernelNodeParams, blockDim) == 20, "block offset");
_Static_assert(offsetof(struct cudaKernelNodeParams, sharedMemBytes) == 32, "shared-memory offset");
_Static_assert(offsetof(struct cudaKernelNodeParams, kernelParams) == 40, "direct-arguments offset");
_Static_assert(offsetof(struct cudaKernelNodeParams, extra) == 48, "extra offset");

/* Explicit direct ABI for CPU fake-library tests. This is not a GPU kernel. */
void fixture_direct_kernel(const float *left, const float *right, unsigned int rows);

int main(void) {
    puts("cudaKernelNodeParams size=56 alignment=8 offsets=0,8,20,32,40,48 pointer_bytes=8");
    return 0;
}
