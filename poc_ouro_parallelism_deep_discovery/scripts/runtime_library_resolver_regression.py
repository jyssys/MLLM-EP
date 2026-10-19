"""CPU-only library-selection regression. No CUDA context or kernel is requested."""
import argparse
import ctypes as C
from pathlib import Path
import traceback
import os
from poc_ouro_parallelism_deep_discovery.contract import write
from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not args.output.exists()
    configure_native_imports()
    import torch
    from poc_ouro_parallelism_deep_discovery.cuda_graph_event_nodes import loaded_cudart_path
    baseline = loaded_cudart_path()
    alternate = '/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12.8.90'
    assert Path(alternate).is_file() and Path(alternate).resolve() != Path(baseline).resolve()
    # Loading an already installed shared library does not execute a GPU kernel.
    second = C.CDLL(alternate, mode=os.RTLD_LOCAL)
    assert second
    relevant = [line for line in Path('/proc/self/maps').read_text().splitlines()
                if '/libcudart.so' in line or '/libtorch_cuda.so' in line]
    record = dict(baseline_runtime=baseline, deliberately_loaded_alternate=alternate,
                  relevant_mapping_lines=relevant, GPU_context_requested=False,
                  CUDA_VISIBLE_DEVICES='', torch_cuda_initialized=torch.cuda.is_initialized(),
                  scope='Reproduce the actual two-library pattern without model loading or GPU execution.')
    assert not record['torch_cuda_initialized']
    try:
        selected = loaded_cudart_path()
        assert Path(selected).resolve() == Path(baseline).resolve(), 'The resolver changed away from the Torch runtime.'
        record.update(status='PASS', selected_runtime=selected)
    except BaseException as error:
        record.update(status='FAIL', error_type=type(error).__name__, traceback=traceback.format_exc())
        write(args.output, record)
        print(record['status'], type(error).__name__, str(error), flush=True)
        raise
    write(args.output, record)
    print('PASS', selected, 'CUDA context initialized:', torch.cuda.is_initialized())


if __name__ == '__main__':
    main()
