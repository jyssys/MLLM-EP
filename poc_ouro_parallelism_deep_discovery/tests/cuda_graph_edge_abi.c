/* CPU ABI regression. No CUDA runtime or driver call. */
#include <stddef.h>
#include <stdio.h>
#include <driver_types.h>

_Static_assert(sizeof(cudaGraphEdgeData) == 8, "edge-data size");
_Static_assert(_Alignof(cudaGraphEdgeData) == 1, "edge-data alignment");
_Static_assert(offsetof(cudaGraphEdgeData, from_port) == 0, "source port offset");
_Static_assert(offsetof(cudaGraphEdgeData, to_port) == 1, "target port offset");
_Static_assert(offsetof(cudaGraphEdgeData, type) == 2, "dependency type offset");
_Static_assert(offsetof(cudaGraphEdgeData, reserved) == 3, "reserved offset");
_Static_assert(sizeof(((cudaGraphEdgeData *)0)->reserved) == 5, "reserved size");
_Static_assert(cudaGraphDependencyTypeDefault == 0, "default dependency");
_Static_assert(cudaGraphDependencyTypeProgrammatic == 1, "programmatic dependency");
_Static_assert(cudaGraphKernelNodePortProgrammatic == 1, "programmatic port");
_Static_assert(cudaGraphKernelNodePortLaunchCompletion == 2, "launch completion port");

int main(void) {
    puts("cudaGraphEdgeData size=8 alignment=1 offsets=0,1,2,3 reserved=5");
    return 0;
}
