"""Separate protocol selection and post-run SDK errors from serving samples."""
import json
import re
from .common import ROOT, digest, write


def context_errors(folder):
    file = folder / 'decode.log'
    if not file.exists():
        return None
    errors = []
    for line in file.read_text(errors='replace').splitlines():
        if 'cuDevicePrimaryCtxGetState' in line and 'unrecognized error code 4' in line:
            match = re.match(r'\[(\d+\.\d+)\]', line)
            errors.append(dict(epoch=float(match.group(1)) if match else None, line=line))
    if not errors:
        return None
    status = json.loads((folder / 'status.json').read_text())
    exits = json.loads((folder / 'worker_exits.json').read_text())
    after = all(row['epoch'] is not None and row['epoch'] > status['time'] for row in errors)
    return dict(name=folder.name, log_SHA256=digest(file), count=len(errors),
                first_error=errors[0], completion_epoch=status['time'],
                first_error_seconds_after_completion=errors[0]['epoch']-status['time'],
                all_after_coordinator_COMPLETE=after, worker_exits=exits,
                classification='POST_RUN_SDK_CONTEXT_TEARDOWN_ERROR' if after else 'ERROR_DURING_OR_BEFORE_COMPLETION')


def main():
    name = 'diagnostic_Native_UCX_protocol_N8_BI1_v1'
    folder = ROOT / 'raw' / name
    assert json.loads((folder / 'dispatcher_result.json').read_text())['complete']
    gate = json.loads((ROOT / 'analysis' / (name+'_output_gate.json')).read_text())
    assert gate['passed']
    file = folder / 'decode.log'
    lines = file.read_text(errors='replace').splitlines()
    excerpts = [dict(line_number=i+1, line=line) for i, line in enumerate(lines)
                if ('remote memory read' in line or ('zero-copy' in line and 'cuda_ipc/cuda' in line)
                    or ('intra-node cfg' in line and 'cuda_ipc/cuda' in line))]
    selected = any('zero-copy' in row['line'] and 'cuda_ipc/cuda' in row['line'] for row in excerpts)
    errors = []
    for candidate in sorted((ROOT / 'raw').glob('Main_Native_*')) + [folder]:
        row = context_errors(candidate)
        if row:
            errors.append(row)
    write(ROOT / 'analysis/UCX_protocol_selection_and_teardown_audit.json', dict(
        name=name, evidence='DIAGNOSTIC_PROTOCOL_SELECTION_AND_TRACE_ANALYSIS',
        output_and_request_lifetime_passed=gate['passed'], decode_log_SHA256=digest(file),
        selected_cuda_ipc_zero_copy=selected, selected_protocol_excerpts=excerpts,
        post_run_context_errors=errors,
        all_recorded_context_errors_after_completion=all(r['all_after_coordinator_COMPLETE'] for r in errors),
        scope='The N8 log reports protocol selection for CUDA-to-CUDA remote reads. '
        'It is not a per-byte DMA trace or a bandwidth measurement. Main was not run with verbose UCX logging. '
        'Post-run context errors remain unresolved even when output, KV credits and normal process exits pass.'))


if __name__ == '__main__':
    main()
