"""CPU-only argument safety controls. No CUDA runtime or driver is loaded."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from poc_ouro_parallelism_deep_discovery.cuda_graph_argument_getter import (
    EXPECTED_ABI, HEADER_HASHES, OPT_IN, Dim3, KernelNodeParams, ReadOnlyArgumentRuntime,
    kernel_abi, pointer_index, validate_policy, verify_headers)


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *arguments):
        return self.callback(*arguments)


def cpu_policy():
    source = Path(__file__).with_name("cuda_graph_argument_abi.c").resolve()
    return {"schema_version": 1, "kernels": {"fixture_direct_kernel": {
        "parameter_count": 3, "direct_pointer_indices": [0, 1],
        "abi_kind": "DIRECT_POINTER_ARGUMENTS_ONLY", "source": {
            "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "locator": "fixture_direct_kernel declaration. CPU fake-library ABI only."}}}}


class FakeLibrary:
    def __init__(self):
        self.nodes = [10, 20]
        self.values = [C.c_uint64(0x11110000), C.c_uint64(0x22220008), C.c_uint32(7)]
        self.arguments = (C.c_void_p * 3)(*(C.addressof(v) for v in self.values))
        self.extra_storage = (C.c_void_p * 1)(0)
        self.events = []
        self.name = b"fixture_direct_kernel"
        self.version = 12090
        self.info_failure = None
        self.info_sizes = [8, 8, 4]
        self.params_failure = 0
        self.name_failure = 0
        self.representation = "direct"
        self.function = 0x4444
        self.get_nodes_count = 0
        self.change_topology = False
        self.cudaRuntimeGetVersion = Function(lambda p: self.put(p, C.c_int, self.version))
        self.cudaGetErrorString = Function(lambda code: b"CPU_FAKE_ERROR")
        self.cudaGraphGetNodes = Function(self.get_nodes)
        self.cudaGraphGetEdges_v2 = Function(lambda graph, a, b, data, count: self.put(count, C.c_size_t, 0))
        self.cudaGraphNodeGetType = Function(lambda node, p: self.put(p, C.c_int, 0 if node == 10 else 4))
        self.cudaGraphKernelNodeGetParams = Function(self.get_params)
        self.cudaFuncGetName = Function(self.get_name)
        self.cudaFuncGetParamInfo = Function(self.get_info)

    @staticmethod
    def put(pointer, kind, value):
        C.cast(pointer, C.POINTER(kind))[0] = value
        return 0

    def get_nodes(self, graph, nodes, count):
        self.get_nodes_count += 1
        current = self.nodes if not self.change_topology or self.get_nodes_count < 3 else [10]
        if nodes is not None:
            for index, node in enumerate(current):
                nodes[index] = node
        return self.put(count, C.c_size_t, len(current))

    def get_params(self, node, output):
        self.events.append(("params", node))
        if self.params_failure:
            return self.params_failure
        params = KernelNodeParams(func=self.function, gridDim=Dim3(1, 2, 3),
                                  blockDim=Dim3(32, 1, 1), sharedMemBytes=128)
        if self.representation != "null":
            params.kernelParams = self.arguments
        if self.representation in {"extra", "mixed"}:
            params.extra = self.extra_storage
            if self.representation == "extra":
                params.kernelParams = C.POINTER(C.c_void_p)()
        C.cast(output, C.POINTER(KernelNodeParams))[0] = params
        return 0

    def get_name(self, output, function):
        self.events.append(("name", function))
        if self.name_failure:
            return self.name_failure
        return self.put(output, C.c_char_p, self.name)

    def get_info(self, function, index, offset, size):
        self.events.append(("info", index))
        if index == self.info_failure or index >= len(self.info_sizes):
            return 1
        self.put(offset, C.c_size_t, [0, 8, 16][index])
        return self.put(size, C.c_size_t, self.info_sizes[index])

    def read(self, address, size):
        self.events.append(("read", size))
        return C.string_at(address, size)


PARAMETERS = {
    "weight.alias0": {"data_ptr": 0x11110000, "storage_data_ptr": 0x11110000},
    "weight.alias1": {"data_ptr": 0x11110000, "storage_data_ptr": 0x11110000},
    "weight.view": {"data_ptr": 0x22220008, "storage_data_ptr": 0x22220000},
}


class ArgumentGetterTests(unittest.TestCase):
    def observe(self, library=None, policy=None, **budgets):
        library = library or FakeLibrary()
        runtime = ReadOnlyArgumentRuntime(library=library, value_reader=library.read)
        result = runtime.observe(123, cpu_policy() if policy is None else policy, PARAMETERS, **budgets)
        return result, library, runtime

    def test_ctypes_abi(self):
        self.assertEqual(kernel_abi(), EXPECTED_ABI)

    def test_compiled_installed_header_abi(self):
        compiler = shutil.which("cc")
        self.assertIsNotNone(compiler)
        include = Path("/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/nvidia/cuda_runtime/include")
        self.assertEqual({name: row["sha256"] for name, row in verify_headers(include).items()}, HEADER_HASHES)
        compiler_include = include.parents[2] / "triton/backends/nvidia/include"
        source = Path(__file__).with_name("cuda_graph_argument_abi.c")
        with tempfile.TemporaryDirectory(prefix="ouro-argument-abi-", dir=source.parent.parent / "artifacts") as directory:
            binary = Path(directory) / "argument_abi"
            built = subprocess.run([compiler, "-std=c11", "-I", str(include), "-I", str(compiler_include),
                                    str(source), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
            self.assertIn("size=56 alignment=8 offsets=0,8,20,32,40,48 pointer_bytes=8", result.stdout)

    def test_import_loads_no_torch_or_cuda(self):
        code = ("import json,sys; import poc_ouro_parallelism_deep_discovery.cuda_graph_argument_getter; "
                "print(json.dumps({'torch': 'torch' in sys.modules, 'cuda': 'libcudart' in "
                "open('/proc/self/maps').read() or 'libcuda.so' in open('/proc/self/maps').read()}))")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), {"torch": False, "cuda": False})

    def test_opt_in_precedes_runtime_resolution(self):
        with mock.patch.dict(os.environ, {OPT_IN: "0"}), mock.patch(
                "poc_ouro_parallelism_deep_discovery.cuda_graph_typed_edges.loaded_cudart_path") as resolve:
            with self.assertRaisesRegex(RuntimeError, OPT_IN):
                ReadOnlyArgumentRuntime()
            resolve.assert_not_called()

    def test_only_read_only_runtime_apis_bound(self):
        _, _, runtime = self.observe()
        self.assertEqual(set(runtime.bound_functions), {"cudaRuntimeGetVersion", "cudaGetErrorString",
            "cudaGraphGetNodes", "cudaGraphGetEdges_v2", "cudaGraphNodeGetType",
            "cudaGraphKernelNodeGetParams", "cudaFuncGetParamInfo", "cudaFuncGetName"})
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            runtime.query("cudaGraphExecKernelNodeSetParams")

    def test_direct_matches_follow_all_metadata_queries(self):
        result, library, _ = self.observe()
        row = result["records"][0]
        self.assertEqual(row["status"], "KNOWN_DIRECT_ABI_OBSERVED")
        self.assertEqual(library.events, [("params", 10), ("name", 0x4444),
            ("info", 0), ("info", 1), ("info", 2), ("read", 8), ("read", 8)])
        self.assertEqual(row["parameters"][0]["matched_parameter_names"], ["weight.alias0", "weight.alias1"])
        self.assertEqual(row["parameters"][1]["matched_parameter_names"], ["weight.view"])
        self.assertEqual(row["parameters"][2]["status"], "METADATA_ONLY")
        self.assertEqual(result["coverage"]["value_reads"], 2)
        self.assertTrue(result["topology_stable"])
        self.assertTrue(result["executable_parameters"].startswith("UNRESOLVED"))
        self.assertTrue(result["executed_memory_reads"].startswith("UNRESOLVED"))
        self.assertEqual(result["timing"], "NOT_MEASURED")
        json.dumps(result)

    def test_exact_data_ptr_only_not_storage_base(self):
        library = FakeLibrary()
        library.values[1].value = 0x22220000
        result, _, _ = self.observe(library)
        self.assertEqual(result["records"][0]["parameters"][1]["status"], "DIRECT_VALUE_NO_DATA_PTR_MATCH")

    def test_empty_policy_skips_unknown_abi_without_metadata_or_values(self):
        result, library, _ = self.observe(policy={"schema_version": 1, "kernels": {}})
        self.assertEqual(result["records"][0]["status"], "UNKNOWN_ABI_UNRESOLVED")
        self.assertFalse(any(event[0] in {"info", "read"} for event in library.events))

    def test_kernel_names_do_not_use_substring_matching(self):
        library = FakeLibrary()
        library.name = b"fixture_direct_kernel_extra"
        result, _, _ = self.observe(library)
        self.assertEqual(result["records"][0]["status"], "UNKNOWN_ABI_UNRESOLVED")

    def test_extra_mixed_and_null_arrays_are_not_read(self):
        for representation, expected in [("extra", "EXTRA_OR_MIXED_ARGUMENT_REPRESENTATION_SKIPPED"),
                ("mixed", "EXTRA_OR_MIXED_ARGUMENT_REPRESENTATION_SKIPPED"),
                ("null", "NULL_DIRECT_ARGUMENT_ARRAY")]:
            with self.subTest(representation=representation):
                library = FakeLibrary()
                library.representation = representation
                result, _, _ = self.observe(library)
                self.assertEqual(result["records"][0]["status"], expected)
                self.assertFalse(any(event[0] in {"info", "read"} for event in library.events))

    def test_failed_metadata_stops_before_any_value_access(self):
        library = FakeLibrary()
        library.info_failure = 1
        result, _, _ = self.observe(library)
        self.assertEqual(result["records"][0]["status"], "PARAMETER_INFO_FAILED_NO_VALUES_READ")
        self.assertEqual([e for e in library.events if e[0] == "info"], [("info", 0), ("info", 1)])
        self.assertFalse(any(event[0] == "read" for event in library.events))

    def test_declared_pointer_size_mismatch_blocks_all_values(self):
        library = FakeLibrary()
        library.info_sizes[1] = 16
        result, _, _ = self.observe(library)
        self.assertEqual(result["records"][0]["status"], "DECLARED_POINTER_SIZE_MISMATCH_NO_VALUES_READ")
        self.assertFalse(any(event[0] == "read" for event in library.events))

    def test_null_argument_storage_is_skipped(self):
        library = FakeLibrary()
        library.arguments[0] = None
        result, _, _ = self.observe(library)
        self.assertEqual(result["records"][0]["parameters"][0]["status"], "NULL_API_ARGUMENT_STORAGE")
        self.assertEqual(result["coverage"]["value_reads"], 1)

    def test_api_failures_and_null_function_are_explicit(self):
        for field, value, status in [("params_failure", 17, "GET_PARAMS_FAILED"),
                ("name_failure", 18, "FUNCTION_NAME_UNRESOLVED"),
                ("function", 0, "NULL_RUNTIME_FUNCTION")]:
            with self.subTest(field=field):
                library = FakeLibrary()
                setattr(library, field, value)
                result, _, _ = self.observe(library)
                self.assertEqual(result["records"][0]["status"], status)
                self.assertFalse(any(e[0] in {"info", "read"} for e in library.events))

    def test_budgets_are_hard_limits(self):
        result, library, _ = self.observe(max_kernel_nodes=0)
        self.assertEqual(result["records"][0]["status"], "KERNEL_BUDGET_EXHAUSTED")
        self.assertEqual(library.events, [])
        result, library, _ = self.observe(max_value_reads=1)
        self.assertEqual(result["coverage"]["value_reads"], 1)
        self.assertEqual(result["records"][0]["parameters"][1]["status"], "VALUE_READ_BUDGET_EXHAUSTED")
        for value in [-1, 8193, True]:
            with self.assertRaises(ValueError):
                self.observe(max_value_reads=value)

    def test_source_hash_and_abi_policy_fail_closed(self):
        for mutate in [lambda p: p["kernels"]["fixture_direct_kernel"].update(parameter_count=65),
                lambda p: p["kernels"]["fixture_direct_kernel"].update(abi_kind="DEVICE_ARGUMENT_BUFFER"),
                lambda p: p["kernels"]["fixture_direct_kernel"].update(direct_pointer_indices=[0, 3]),
                lambda p: p["kernels"]["fixture_direct_kernel"]["source"].update(sha256="0" * 64)]:
            policy = cpu_policy()
            mutate(policy)
            with self.assertRaises(ValueError):
                validate_policy(policy)

    def test_wrong_runtime_version_rejected(self):
        library = FakeLibrary()
        library.version = 12080
        with self.assertRaisesRegex(RuntimeError, "12.9"):
            ReadOnlyArgumentRuntime(library=library)

    def test_topology_change_is_not_silently_accepted(self):
        library = FakeLibrary()
        library.change_topology = True
        with self.assertRaisesRegex(RuntimeError, "topology changed"):
            self.observe(library)

    def test_explicit_tensor_pointer_required(self):
        for record in [{"storage_data_ptr": 123}, {"data_ptr": 0}, {"data_ptr": True}]:
            with self.assertRaises(ValueError):
                pointer_index({"weight": record})


if __name__ == "__main__":
    unittest.main()
