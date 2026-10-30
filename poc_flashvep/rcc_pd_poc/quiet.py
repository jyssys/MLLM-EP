"""Three independent starts per arm. Abort a model on unexplained output drift."""
import subprocess
from .common import ROOT, setup, write
from .discovery import launch
from .analyze import read_rows


def main():
    setup()
    results = []
    for model in ('ouro', 'huginn'):
        reference = None
        failed = False
        for repeat in range(3):
            arms = ['PD_G1', 'A_sep', 'A_native_mixed']
            if repeat % 2:
                arms.reverse()
            for arm in arms:
                cfg = dict(name=f'quiet128_v1_{model}_{arm}_{repeat}', model=model, arm=arm,
                           gpus=[0,1], dataset='prefix_128.json', count=8, cap=32, chunk=128,
                           capacity=8, arrival_offsets=[i*(.04 if model == 'ouro' else .15) for i in range(8)],
                           seed=2026103017, graphs=True, graph_rows=8, warmup=True,
                           probe=False, verify_kv=False, profile=False, target_trace=False,
                           lane='quiet_pair', timeout=900)
                processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                                      '--format=csv,noheader'], text=True).strip()
                if processes:
                    raise RuntimeError('The quiet lane is not empty before an independent start: '+processes)
                result = launch(cfg)
                rows = read_rows(ROOT/'raw'/cfg['name']) if result.get('exit_code', 0) == 0 else []
                keyed = {r['sample_id']:(r['input_ids'], r['output_ids'], r['exit_depths'], r['finish_reason']) for r in rows}
                if reference is None and rows:
                    reference = keyed
                exact = sum(value == reference.get(index) for index, value in keyed.items()) if reference else 0
                result.update(tokens_exact=exact, N=len(rows), model=model)
                results.append(result)
                write(ROOT/'state/quiet_validation.json', dict(status='RUNNING', results=results))
                if len(rows) != 8 or exact != 8:
                    failed = True
                    break
            if failed:
                break
    write(ROOT/'state/quiet_validation.json', dict(status='FINITE_PILOT_PASS' if all(r['tokens_exact'] == r['N'] == 8 for r in results) else 'BLOCKED_CORRECTNESS',
                                                  results=results, limitation='N8 mechanism cohorts. No benchmark accuracy or capacity claim.'))


if __name__ == '__main__':
    main()
