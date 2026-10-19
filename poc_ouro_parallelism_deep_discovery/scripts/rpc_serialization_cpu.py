"""Exercise the installed native RPC encoder without a model or CUDA context."""
import argparse
import os
from poc_ouro_parallelism_deep_discovery.contract import ROOT, sha, write
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def diagnostic_function(worker):
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = ROOT / args.output
    if path.exists():
        raise FileExistsError(path)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    assert os.environ.get('VLLM_ALLOW_INSECURE_SERIALIZATION', '0') == '0'
    configure_native_imports()
    import torch
    import vllm.v1.serial_utils as serial
    assert not torch.cuda.is_initialized()
    encoder = serial.MsgpackEncoder()
    try:
        encoder.encode((0, 17, 'collective_rpc', (diagnostic_function, 120, (), {})))
    except TypeError as exc:
        function = dict(status='REJECTED_AS_EXPECTED', type=type(exc).__name__, error=str(exc))
    else:
        raise AssertionError('Unexpected callable acceptance with insecure fallback disabled')
    value = (0, 17, 'collective_rpc', ('inspect_guard_identity', 120, (),
                                    dict(expected={'LOOPED_KV_OUTPUT': 'CPU_PUBLIC_FIXTURE'})))
    encoded = encoder.encode(value)
    assert encoded and not torch.cuda.is_initialized()
    write(path, dict(function_case=function, string_case='ENCODED_WITHOUT_PICKLE_FALLBACK',
          insecure_serialization_enabled=False, CUDA_context_created=False,
          serial_utils_path=serial.__file__, serial_utils_sha256=sha(serial.__file__),
          scope='Actual installed frontend encoder. No native model or GPU execution.'))
    print('RPC_ENCODER_CPU: callable rejected, method string accepted, insecure fallback OFF')


if __name__ == '__main__':
    main()
