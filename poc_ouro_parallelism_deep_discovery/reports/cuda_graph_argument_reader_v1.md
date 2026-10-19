# Bounded graph argument reader

## Result and limit

The opt-in reader and guarded runner are ready for source review.
The CPU controls pass 35 tests under each of two Python interpreters.
No GPU test ran.
No live CUDA API call ran.
This work creates no real-kernel ABI policy.

The reader does not establish cached-decode weight identity.
An empty policy exports kernel names, launch metadata, representation status, and coverage.
It reads no argument values.
Unknown kernels and opaque library GEMMs remain `UNRESOLVED`.

The [source contract](cuda_graph_argument_feasibility_v1.md) records the official APIs, installed headers, hashes, and ownership limits.
The implementation uses `cudaGraphKernelNodeGetParams`, `cudaFuncGetParamInfo`, and `cudaFuncGetName`.
It never converts a runtime function into CUfunction.

## Safety gate

The live constructor requires `OURO_CUDA_GRAPH_ARGUMENT_DIAGNOSTIC=1`.
It reuses the existing Torch-linked runtime selection with `RTLD_NOLOAD`.
It requires CUDA 12.9 and the pinned installed-header hashes.
The reader binds only getters, runtime identification, and error reporting.

A value read requires all these conditions:

1. The caller retains the source graph and serializes graph access.
2. The node returns only the direct `kernelParams` representation.
3. An explicit policy matches the exact kernel name.
4. The policy supplies a declared count, direct-pointer indices, and a source file with its hash and locator.
5. Every declared index succeeds in `cudaFuncGetParamInfo` before any argument-array access.
6. Each selected argument has an API-reported size of eight bytes.
7. The node and value-read budgets permit the read.

The reader copies eight bytes from the node-owned host argument storage.
It does not add the device-layout offset to that separate host pointer.
It does not follow the resulting value.
It compares that value only with explicit model-tensor `data_ptr` records.
It does not infer pointer types from size alone.

The policy permits at most 64 declared arguments per kernel.
The observation permits at most 4,096 kernel nodes and 8,192 value reads.
Metadata failures stop inspection of that node before any argument-value read.
The reader skips `extra`, mixed representations, unknown ABIs, and null storage.
Policy validation rejects declared indirect layouts.
It does not inspect child graphs, descriptors, pointer arrays, or device argument buffers.

A source hash does not prove correspondence between source and a loaded binary.
Policy review must establish that correspondence before the operator uses a real policy.
The current implementation supplies no such policy.

## Evidence scope

A match has status `TEMPLATE_ARGUMENT_VALUE_MATCH`.
It establishes only equality between a retained source-template argument value and a recorded tensor address.
It does not prove weight-operand semantics, executed memory reads, or HBM traffic.
It does not prove that executable-only updates preserved those arguments.

The worker records separate source-graph and executable handles.
It checks source topology and handle stability.
These checks do not establish argument-value stability or an executable no-update chain.
The output keeps executed operand identity `UNRESOLVED`.

The worker reuses the existing retained-graph debug process and actual replay identities.
The runner keeps native asynchronous scheduling and the selected source arm's R4 configuration.
It requires the existing GPU guard and new isolated cache directories.
It performs three warmup requests and one observed four-token teacher-forced request if authorized later.
Teacher forcing is a workload control, not a natural-quality result.
Debug retention also excludes this lane from timing claims.

## CPU validation

The C fixture compiles against the pinned CUDA 12.9 headers.
Static checks cover structure size, alignment, offsets, and all three function prototypes.
The fixture calls no CUDA function and links no CUDA runtime.

Fake-library tests cover direct matches, tensor views, aliases, failed queries, null storage, and fixed read budgets.
They also cover unknown names, private ABI exclusion, source-hash mismatch, and topology changes.
Import tests check that the new core and runner load neither Torch nor CUDA.
The worker's missing-opt-in test stops before the existing Torch-based debug module imports.

Each interpreter passes 25 new tests and 10 existing typed-edge regression tests.
Each interpreter repeats the same software controls.
The tests are not 70 independent model experiments.
The [default-interpreter record](../artifacts/cuda_graph_argument_cpu_validation_v1.json) contains the command, full output, timestamps, and hashes.
The [native-interpreter record](../artifacts/cuda_graph_argument_cpu_validation_native_v1.json) supplies the same metadata for the native Python 3.14 environment.
Both records show unchanged existing diagnostic files across validation.

## Files and handoff

| File | Role |
|---|---|
| `cuda_graph_argument_getter.py` | CPU-importable read-only core and explicit ABI policy |
| `graph_argument_probe.py` | Retained-graph worker extension |
| `scripts/native_graph_argument_probe.py` | Guarded native, alias, or copies diagnostic |
| `scripts/cuda_graph_argument_cpu_validation.py` | Reproducible CPU validation record |
| `tests/test_cuda_graph_argument_getter.py` | Argument safety controls |
| `tests/test_native_graph_argument_probe_cpu.py` | Runner and import controls |
| `tests/cuda_graph_argument_abi.c` | Installed-header ABI checks |

Main review must precede any guarded GPU probe.
Do not weaken the ABI gate to obtain matches.
Without a source-backed direct signature, use the empty-policy inventory or stop with unresolved coverage.
The [implementation ledger](../artifacts/cuda_graph_argument_reader_v1.json) records actual work intervals.
It excludes the compaction gap and does not add CPU validation intervals twice.
