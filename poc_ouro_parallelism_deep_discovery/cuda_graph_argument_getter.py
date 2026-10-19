"""Bounded source-graph argument observations. Import does not load CUDA."""
from collections import Counter, defaultdict
import ctypes as C
import hashlib
import os
from pathlib import Path
import sys

from poc_ouro_parallelism_deep_discovery.cuda_graph_typed_edges import ReadOnlyGraphRuntime


OPT_IN = "OURO_CUDA_GRAPH_ARGUMENT_DIAGNOSTIC"
HEADER_HASHES = {
    "cuda_runtime_api.h": "e88147ea668332c16fba1614f159c105973cec9993870e09a4217dd7c17c8f02",
    "driver_types.h": "537c2198bc3a320a66ac97623283f50bb61e1f25caa317c9994f3738a85b799b",
}
MAX_PARAMETERS = 64


class Dim3(C.Structure):
    _fields_ = [("x", C.c_uint), ("y", C.c_uint), ("z", C.c_uint)]


class KernelNodeParams(C.Structure):
    _fields_ = [("func", C.c_void_p), ("gridDim", Dim3), ("blockDim", Dim3),
                ("sharedMemBytes", C.c_uint), ("kernelParams", C.POINTER(C.c_void_p)),
                ("extra", C.POINTER(C.c_void_p))]


def kernel_abi():
    return dict(size=C.sizeof(KernelNodeParams), alignment=C.alignment(KernelNodeParams),
                pointer_bytes=C.sizeof(C.c_void_p), byteorder=sys.byteorder,
                offsets={name: getattr(KernelNodeParams, name).offset
                         for name, _ in KernelNodeParams._fields_})


EXPECTED_ABI = dict(size=56, alignment=8, pointer_bytes=8, byteorder="little",
    offsets=dict(func=0, gridDim=8, blockDim=20, sharedMemBytes=32, kernelParams=40, extra=48))


def verify_headers(include):
    """Check the exact installed headers used by the CPU ABI regression."""
    rows = {}
    for name, expected in HEADER_HASHES.items():
        path = Path(include) / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"The pinned CUDA 12.9 header differs: {path}")
        rows[name] = dict(path=str(path), sha256=actual)
    return rows


def validate_policy(policy):
    """Validate an explicit direct-argument ABI policy and its source hash."""
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "kernels"}:
        raise ValueError("The kernel policy requires schema_version and kernels only.")
    if policy["schema_version"] != 1 or not isinstance(policy["kernels"], dict):
        raise ValueError("The kernel policy schema is not supported.")
    if len(policy["kernels"]) > 256:
        raise ValueError("The kernel policy exceeds 256 exact names.")
    result = {}
    required = {"parameter_count", "direct_pointer_indices", "abi_kind", "source"}
    for name, entry in policy["kernels"].items():
        if not isinstance(name, str) or not name or len(name) > 8192 or "\x00" in name:
            raise ValueError("A kernel policy name is invalid.")
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("Each policy entry requires an explicit direct ABI and source.")
        count, indices = entry["parameter_count"], entry["direct_pointer_indices"]
        if type(count) is not int or not 1 <= count <= MAX_PARAMETERS:
            raise ValueError("A declared parameter count is outside 1..64.")
        if not isinstance(indices, list) or not indices or any(type(i) is not int for i in indices):
            raise ValueError("Direct pointer indices must be a nonempty integer list.")
        if len(set(indices)) != len(indices) or any(not 0 <= i < count for i in indices):
            raise ValueError("Direct pointer indices must be unique and within the declared count.")
        if entry["abi_kind"] != "DIRECT_POINTER_ARGUMENTS_ONLY":
            raise ValueError("Indirect, structure, pointer-array, and unknown ABIs are not supported.")
        source = entry["source"]
        if not isinstance(source, dict) or set(source) != {"path", "sha256", "locator"}:
            raise ValueError("The ABI source requires path, sha256, and locator.")
        if not all(isinstance(source[k], str) and source[k] for k in source):
            raise ValueError("The ABI source fields must contain strings.")
        path = Path(source["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("The ABI source must name an explicit existing absolute file.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source["sha256"]:
            raise ValueError("The ABI source hash differs from the policy.")
        result[name] = dict(entry, direct_pointer_indices=sorted(indices),
                            source=dict(source, path=str(path.resolve())))
    return {"schema_version": 1, "kernels": result}


def pointer_index(parameters):
    """Index explicit tensor data_ptr values. Do not match storage bases or ranges."""
    result = defaultdict(list)
    if not isinstance(parameters, dict):
        raise ValueError("The parameter snapshot must be a dictionary.")
    for name, metadata in parameters.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise ValueError("Each parameter requires a name and metadata dictionary.")
        value = metadata.get("data_ptr")
        if type(value) is not int or not 0 < value < 2**64:
            raise ValueError("Each parameter requires an explicit nonzero uint64 data_ptr.")
        result[value].append(name)
    return {value: sorted(names) for value, names in result.items()}


class ReadOnlyArgumentRuntime(ReadOnlyGraphRuntime):
    """Read known source-node arguments. Never query executable parameters or device memory."""
    def __init__(self, library=None, value_reader=None):
        if kernel_abi() != EXPECTED_ABI:
            raise RuntimeError("The host kernel-node ABI differs from the pinned CUDA 12.9 ABI.")
        if library is None and os.environ.get(OPT_IN) != "1":
            raise RuntimeError(f"Set {OPT_IN}=1 before the live read-only diagnostic.")
        super().__init__(library=library)
        self.header_contract = (verify_headers(Path(self.path).resolve().parent.parent / "include")
                                if library is None else {"status": "CPU_TEST_DOUBLE"})
        bindings = {
            "cudaGraphKernelNodeGetParams": [C.c_void_p, C.POINTER(KernelNodeParams)],
            "cudaFuncGetParamInfo": [C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)],
            "cudaFuncGetName": [C.POINTER(C.c_char_p), C.c_void_p],
        }
        self.bound_functions.update(bindings)
        for name, arguments in bindings.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, C.c_int
        if value_reader is not None and library is None:
            raise ValueError("A custom value reader is permitted only for CPU test doubles.")
        self.value_reader = value_reader or C.string_at

    def query(self, name, *arguments):
        if name not in self.bound_functions:
            raise RuntimeError("The read-only runtime refuses an unbound API.")
        code = int(getattr(self.lib, name)(*arguments))
        return code, None if code == 0 else repr(self.lib.cudaGetErrorString(code))

    def observe(self, graph, policy, parameters, *, max_kernel_nodes=4096, max_value_reads=8192):
        """Observe one retained, externally serialized source graph under fixed budgets."""
        if type(max_kernel_nodes) is not int or not 0 <= max_kernel_nodes <= 4096:
            raise ValueError("The kernel-node budget must be within 0..4096.")
        if type(max_value_reads) is not int or not 0 <= max_value_reads <= 8192:
            raise ValueError("The value-read budget must be within 0..8192.")
        policy = validate_policy(policy)
        pointers = pointer_index(parameters)
        before = self.snapshot(graph)
        rows, value_reads, inspected = [], 0, 0
        for node in before["nodes"]:
            row = dict(node_handle=node["handle"], node_kind=node["kind"])
            rows.append(row)
            if node["kind"] != 0:
                row["status"] = "NON_KERNEL_NODE_NOT_INSPECTED"
                continue
            if inspected >= max_kernel_nodes:
                row["status"] = "KERNEL_BUDGET_EXHAUSTED"
                continue
            inspected += 1
            params = KernelNodeParams()
            code, error = self.query("cudaGraphKernelNodeGetParams", int(node["handle"], 16), C.byref(params))
            if code:
                row.update(status="GET_PARAMS_FAILED", error_code=code, error=error)
                continue
            row.update(runtime_function=hex(params.func or 0),
                       grid=[params.gridDim.x, params.gridDim.y, params.gridDim.z],
                       block=[params.blockDim.x, params.blockDim.y, params.blockDim.z],
                       shared_memory_bytes=params.sharedMemBytes)
            if not params.func:
                row["status"] = "NULL_RUNTIME_FUNCTION"
                continue
            name = C.c_char_p()
            code, error = self.query("cudaFuncGetName", C.byref(name), params.func)
            if code or not name.value:
                row.update(status="FUNCTION_NAME_UNRESOLVED", error_code=code, error=error)
                continue
            kernel_name = name.value.decode("utf-8", errors="backslashreplace")
            row["kernel_name"] = kernel_name
            if params.extra:
                row["status"] = "EXTRA_OR_MIXED_ARGUMENT_REPRESENTATION_SKIPPED"
                continue
            if not params.kernelParams:
                row["status"] = "NULL_DIRECT_ARGUMENT_ARRAY"
                continue
            entry = policy["kernels"].get(kernel_name)
            if entry is None:
                row["status"] = "UNKNOWN_ABI_UNRESOLVED"
                continue
            row["policy"] = entry
            metadata, failed = [], False
            # Query all declared indices before any argument-array access or value read.
            for index in range(entry["parameter_count"]):
                offset, size = C.c_size_t(), C.c_size_t()
                code, error = self.query("cudaFuncGetParamInfo", params.func, index,
                                         C.byref(offset), C.byref(size))
                item = dict(index=index, error_code=code)
                metadata.append(item)
                if code:
                    item.update(status="PARAMETER_INFO_FAILED", error=error)
                    failed = True
                    break
                item.update(device_layout_offset=offset.value, size_bytes=size.value,
                            status="METADATA_ONLY")
            row["parameters"] = metadata
            if failed:
                row["status"] = "PARAMETER_INFO_FAILED_NO_VALUES_READ"
                continue
            direct = set(entry["direct_pointer_indices"])
            if any(metadata[i]["size_bytes"] != 8 for i in direct):
                row["status"] = "DECLARED_POINTER_SIZE_MISMATCH_NO_VALUES_READ"
                continue
            row["status"] = "KNOWN_DIRECT_ABI_OBSERVED"
            for item in metadata:
                index = item["index"]
                if index not in direct:
                    continue
                if value_reads >= max_value_reads:
                    item["status"] = "VALUE_READ_BUDGET_EXHAUSTED"
                    continue
                # The API confirms this index and size. The source declares a direct pointer.
                # Each array entry points to separate value bytes. Do not add layout offsets.
                argument_storage = params.kernelParams[index]
                if not argument_storage:
                    item["status"] = "NULL_API_ARGUMENT_STORAGE"
                    continue
                raw = self.value_reader(argument_storage, 8)
                if not isinstance(raw, bytes) or len(raw) != 8:
                    raise RuntimeError("The bounded argument reader did not return exactly eight bytes.")
                value_reads += 1
                value = int.from_bytes(raw, "little")
                matches = pointers.get(value, [])
                item.update(status="TEMPLATE_ARGUMENT_VALUE_MATCH" if matches else "DIRECT_VALUE_NO_DATA_PTR_MATCH",
                            value_u64=value, value_hex=hex(value), matched_parameter_names=matches,
                            evidence_scope="SOURCE_TEMPLATE_ARGUMENT_ONLY")
        after = self.snapshot(graph)
        if before["fingerprint"] != after["fingerprint"]:
            raise RuntimeError("The source-graph topology changed during read-only inspection.")
        return dict(schema_version=1, graph_handle=before["graph_handle"],
            runtime_library=self.path, runtime_version=self.version, abi=kernel_abi(),
            header_contract=self.header_contract, policy=policy, records=rows,
            coverage=dict(top_level_nodes=len(rows), kernel_nodes=sum(n["kind"] == 0 for n in before["nodes"]),
                          inspected_kernel_nodes=inspected, value_reads=value_reads,
                          explicit_tensor_names=len(parameters), explicit_data_ptrs=len(pointers),
                          node_status_counts=dict(Counter(r["status"] for r in rows)),
                          argument_status_counts=dict(Counter(p["status"] for r in rows for p in r.get("parameters", [])))),
            budgets=dict(max_kernel_nodes=max_kernel_nodes, max_parameters_per_known_kernel=MAX_PARAMETERS,
                         max_value_reads=max_value_reads),
            topology_fingerprint_before=before["fingerprint"], topology_fingerprint_after=after["fingerprint"],
            topology_stable=True, parameter_value_stability="NOT_ESTABLISHED_BY_TOPOLOGY_CHECK",
            graph_access_contract="Caller retains source graph and serializes all graph access.",
            policy_limit="A source hash and exact name do not prove binary-to-source correspondence or operand semantics.",
            executable_parameters="UNRESOLVED. Source template is not a cudaGraphExec parameter getter.",
            executed_memory_reads="UNRESOLVED. A value match does not prove an executed weight read.",
            opaque_library_gemms="UNRESOLVED unless an exact direct ABI policy covers that kernel.",
            scope="Top-level known direct arguments only. No indirect, extra, child, or device-memory reads.",
            evidence_class="SOURCE_TEMPLATE_ARGUMENT_OBSERVATION", timing="NOT_MEASURED", mutation="NONE")
