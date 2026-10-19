# Read-only CUDA Graph argument feasibility

## Decision

CUDA 12.9 supports bounded inspection of captured kernel argument values under specific conditions.
It does not provide a universal decoder for every captured library GEMM's weight operand.
The current prefill metadata observation therefore remains separate from cached-decode kernel pointer identity.

This audit invokes no CUDA API, loads no CUDA library, and reads no unknown host or device pointer.
It inspects documentation, installed headers, and ELF symbols only.
It changes no diagnostic module or runner.

## Available APIs

| API | What it supplies | What it does not supply |
|---|---|---|
| `cudaGraphKernelNodeGetParams` | Source-node function, launch configuration, and node-owned argument representation | Argument count, types, operand names, or current executable-only updates |
| `cudaFuncGetParamInfo` | One parameter's device-layout offset and byte size | C/C++ type or nested structure fields |
| `cuFuncGetParamInfo` | Equivalent information for a valid CUfunction | Permission to reinterpret a runtime host stub as CUfunction |
| `cuKernelGetParamInfo` | Equivalent information for a valid CUkernel | A typed weight-operand description |

No separate kernel-parameter count getter appears in the inspected 12.9 headers.
`kernelParams` is not documented as null-terminated.
An implementation must not discover its length through memory scanning.
Known signatures can supply N.
Bounded API-only index queries can inventory successful indices without reading unknown array entries.
A failed query must end inspection, not authorize a guessed size or a fabricated complete count.

The [runtime execution API](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__EXECUTION.html) documents offset/size queries.
The [driver execution API](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__EXEC.html) documents CUfunction queries.

## Safe value access and unsafe interpretation

The [graph getter contract](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__GRAPH.html) gives ownership and lifetime for returned argument arrays and values.
Those values remain valid until node destruction or parameter modification.
Inspection must retain the graph and serialize all access to it.
NVIDIA also requires serialization for apparently read-only graph APIs.
See [graph thread safety](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/graphs-thread-safety.html).

For `kernelParams`, each entry points to that argument's value bytes.
The queried device-layout offset is not an offset to add to that separate value pointer.
For `extra`, only the documented tagged buffer format supplies a buffer address and buffer size.
An offset/size slice must remain within that declared buffer.
Unknown tags or representations require an unresolved result.

| Within a bounded, future authorized diagnostic | Outside the safe boundary |
|---|---|
| Copy the documented number of bytes from a successful node-owned argument entry | Scan argument arrays or arbitrary address ranges |
| Decode a direct pointer using a verified kernel signature | Treat every eight-byte value as a pointer |
| Compare that pointer with known tensor allocation ranges and view offsets | Dereference a candidate device pointer as host memory |
| Inspect a known by-value structure using its exact ABI | Search unknown structures or TMA descriptors for pointer-shaped words |
| Report a device argument-buffer pointer as indirect | Follow that pointer without its layout, extent, and stable version |

A pointer-sized parameter can also be an integer or an opaque handle.
A GEMM can use a by-value structure, device argument buffer, pointer array, workspace, or opaque descriptor.
The public cuBLAS call signature does not define its private captured kernel's argument layout.
Pointer equality in an untyped byte search would be a candidate match, not proof of the weight operand.
Packed weights and tensor views also require explicit address-range and offset accounting.

## Function handles must match the API

A runtime kernel entry symbol is not interchangeable with an arbitrary CUfunction handle.
The documented `cudaGetFuncBySymbol` conversion returns `cudaFunction_t`, which interoperates with CUfunction.
`cudaGetKernel` returns `cudaKernel_t`, which interoperates with CUkernel.
See [runtime/driver interactions](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__DRIVER.html).
Runtime symbols also require the matching runtime registration instance.

The driver graph getter returns a CUfunction or a CUkernel through its versioned structure.
In the installed header, its macro resolves to `cuGraphKernelNodeGetParams_v2`.
That structure has additional kernel/context fields beyond the runtime structure.
Using the wrong foreign-function structure risks memory corruption.
A header-typed native wrapper would avoid guessing this ABI.
This source audit implements no wrapper.

## Source graph versus replayed executable

The getter accepts a graph node, not a `cudaGraphExec_t`.
The inspected APIs contain executable setters but no matching executable kernel-parameter getter.
An executable update can change future launches without changing the original source node.
Thus, source-node values prove replay operands only with matching instantiation and an audited no-update chain.
Indirect device argument contents need additional version evidence.

Installed PyTorch 2.11 also matters.
Its `CUDAGraph` documentation states that default `keep_graph=False` destroys the source graph after instantiation.
It describes `keep_graph=True` as the route to retain `raw_cuda_graph`.
This audit does not change capture settings or establish which retained graph the current runner exposes.
It does not reinterpret an executable handle as a graph handle.

## Installed-version evidence and next decision

The relevant Python package is `nvidia-cuda-runtime-cu12` version `12.9.79`.
Its headers declare `CUDA_VERSION=CUDART_VERSION=12090`.
Its `libcudart.so.12` exports the runtime getter, parameter-info query, and documented conversion functions.
Static symbol presence does not prove which runtime instance a live worker loaded.

`/usr/local/cuda` instead resolves to CUDA 12.8.
It must not be mislabeled as the 12.9 header source.
The [structured observations](../artifacts/cuda_graph_argument_feasibility_v1.json) record exact paths, hashes, locators, and symbol evidence.

The smallest next decision is per-kernel feasibility, not a large profiler.
Require a retained source graph, matched executable provenance, correct function handle, known argument extent, and a verified operand layout.
If any condition fails, label that kernel `POINTER_BINDING_UNRESOLVED`.
Do not use unknown-pointer reads, graph mutation, or changed kernels to force a result.
Even complete operand identity would not measure HBM traffic or establish a causal speedup from weight sharing.

Work began at 2026-10-19 14:50:02 UTC.
The structured observations record the segment end after report validation.
No GPU or memory-inspection test ran.
