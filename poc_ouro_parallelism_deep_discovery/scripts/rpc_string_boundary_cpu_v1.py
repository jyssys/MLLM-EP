"""Check native RPC serialization and string dispatch without a model worker."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from poc_ouro_parallelism_deep_discovery.contract import ROOT, sha, write
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Require a new artifact path inside this campaign.')
    assert 'torch' not in sys.modules
    assert os.environ.get('VLLM_ALLOW_INSECURE_SERIALIZATION', '0') == '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    configure_native_imports()
    import torch
    import vllm.envs as envs
    import vllm.v1.serial_utils as serial
    from vllm.v1.engine import UtilityOutput
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from poc_ouro_parallelism_deep_discovery.hybrid_cohort_probe import HybridCohortProbe
    from poc_ouro_parallelism_deep_discovery.process_lease_diagnostics import native_worker_snapshot

    assert not envs.VLLM_ALLOW_INSECURE_SERIALIZATION
    assert not torch.cuda.is_initialized()
    native = Path(serial.__file__).parents[1]
    paths = [Path(__file__).resolve(), ROOT / 'hybrid_cohort_probe.py',
             ROOT / 'process_lease_diagnostics.py', Path(serial.__file__),
             native / 'v1/engine/core_client.py', native / 'v1/worker/worker_base.py',
             native / 'v1/executor/multiproc_executor.py']
    source_hashes = {str(path): sha(path) for path in paths}
    start = datetime.now(timezone.utc).isoformat()
    expected = dict(LOOPED_KV_OUTPUT='CPU_PUBLIC_GUARD', CUDA_VISIBLE_DEVICES='',
                    VLLM_WORKER_MULTIPROC_METHOD='spawn')
    function_payload = (0, 123, 'collective_rpc',
                        (native_worker_snapshot, 120, (), dict(expected=expected)))
    try:
        serial.MsgpackEncoder().encode(function_payload)
    except TypeError as error:
        rejection = dict(type=type(error).__name__, error=str(error))
    else:
        raise AssertionError('The function bypassed the expected serialization guard.')
    payload = (0, 123, 'collective_rpc',
               ('inspect_guard_identity', 120, (), dict(expected=expected)))
    encoded = serial.MsgpackEncoder().encode(payload)
    decoded = serial.MsgpackDecoder().decode(encoded)
    method, timeout, arguments, keywords = decoded[3]
    assert method == 'inspect_guard_identity' and timeout == 120
    assert keywords == dict(expected=expected)
    wrapper = WorkerWrapperBase(rpc_rank=0)
    wrapper.worker = HybridCohortProbe()
    result = getattr(wrapper, method)(*arguments, **keywords)
    assert result['pid'] == os.getpid() and result['diagnostic_only']
    response = UtilityOutput(call_id=123, result=serial.UtilityResult([result]))
    decoded_response = serial.MsgpackDecoder(UtilityOutput).decode(
        serial.MsgpackEncoder().encode(response))
    assert decoded_response.call_id == 123 and decoded_response.result.result == [result]
    assert not envs.VLLM_ALLOW_INSECURE_SERIALIZATION
    assert not torch.cuda.is_initialized()
    record = dict(start_utc=start, end_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, '-m',
                 'poc_ouro_parallelism_deep_discovery.scripts.rpc_string_boundary_cpu_v1',
                 '--output', str(output)], source_sha256_before_run=source_hashes,
        function_rejected=rejection, string_request_roundtrip=True,
        encoded_request_frames=len(encoded), encoded_request_bytes=sum(map(len, encoded)),
        actual_worker_wrapper_string_dispatch=True, response_roundtrip=True,
        insecure_serialization_enabled=False, cuda_initialized=False,
        scope='CPU serializer and bound wrapper dispatch only. No socket transport, model worker initialization, engine startup, or process lease proof.')
    write(output, record)
    print('RPC_STRING_BOUNDARY_CPU: function rejected, string request and response roundtrip PASS, wrapper dispatch PASS, insecure fallback OFF')


if __name__ == '__main__':
    main()
