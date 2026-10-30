"""Check complete evidence and preserve the pre-existing Git index."""
import csv
import hashlib
import json
import subprocess
from .common import ROOT, REPO, OWNED, write


def main():
    rows = list(csv.DictReader((ROOT/'full_results.csv').open()))
    assert len(rows) == 36, len(rows)
    assert all(r['work_exact'] == 'True' and int(r['requests']) == 8 for r in rows)
    comparisons = json.loads((ROOT/'correctness_matrix.json').read_text())
    assert comparisons and all(r['tokens_exact']==r['inputs_exact']==r['depths_exact']==r['stop_exact']==r['N']==8 for r in comparisons)
    states = json.loads((ROOT/'prompt_state_comparisons.json').read_text())
    assert len(states) == 16 and all(r['hidden_exact'] and r['KV_planes_exact'] for r in states)
    quiet = json.loads((ROOT/'state/quiet_validation.json').read_text())
    assert quiet['status'] == 'FINITE_PILOT_PASS' and len(quiet['results']) == 18
    outside = []
    fabric_samples = 0
    for p in (ROOT/'raw').glob('quiet*/fabric_process_samples.jsonl'):
        for line in p.read_text().splitlines():
            value = json.loads(line)
            fabric_samples += 1
            outside.extend(x for x in value['processes'] if x['outside_assigned_pair'])
    assert fabric_samples and not outside
    evidence = json.loads((ROOT/'operator_fusion_evidence.json').read_text())
    counts = {r['run'].split('_')[2]:r['GEMM_GEMV_kernel_events'] for r in evidence['selected_kernel_traces']}
    assert counts == {'ouro':337, 'huginn':52}, counts
    index = subprocess.check_output(['git','ls-files','--stage'],cwd=REPO)
    kept = b''.join(line for line in index.splitlines(keepends=True)
                    if not any(line.split(b'\t',1)[-1].decode().strip().startswith(prefix) for prefix in OWNED))
    expected = json.loads((ROOT/'configs/cpu_audit.json').read_text())['initial_git_index_sha256']
    actual = hashlib.sha256(kept).hexdigest()
    assert actual == expected, (actual,expected)
    planned = []
    for model in ('ouro','huginn'):
        for budget in (2,4,8):
            for workload in ('W1_LG','W1_NAT','W2','W3','W4'):
                planned.append(dict(model=model,GPU_budget=budget,workload=workload,
                                    status='NOT_RUN',reason='Stage-1 stop: existing A/Native Mixed already implements the proposed fusion. No admission or scale-out GO.'))
    with (ROOT/'unmeasured_serving_matrix.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(planned[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(planned)
    matrix = []
    for r in rows:
        cfg=json.loads((ROOT/'raw'/r['name']/'rcc_contract.json').read_text())
        matrix.append(dict(run=r['name'],model=r['model'],arm=r['arm'],physical_GPUs=r['gpus'],
                           GPU_budget=2,prompt_tokens=r['input_tokens'],output_cap=cfg['cap'],
                           graphs=cfg['graphs'],probe=cfg['probe'],state_probe=cfg.get('state_probe',False),
                           seed=cfg['seed'],evidence=r['evidence'],status='COMPLETE'))
    with (ROOT/'experiment_matrix.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(matrix[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(matrix)
    write(ROOT/'validation.json',dict(status='FINITE_MECHANISM_EVIDENCE_VALIDATED',runs=len(rows),
          measured_requests=sum(int(r['requests']) for r in rows),
          measured_output_tokens=sum(int(r['output_tokens']) for r in rows),
          same_depth_row_accounting_passes=len(rows),quiet_runs=18,
          quiet_fabric_samples=fabric_samples,outside_pair_samples=0,
          selected_physical_GEMM_kernel_counts=counts,
          prior_git_index_sha256=expected,preserved_git_index_sha256=actual,
          scope='Finite mechanism evidence. No original-HF lossless, benchmark accuracy, SLO-goodput, or capacity claim.'))
    print(json.loads((ROOT/'validation.json').read_text()))


if __name__=='__main__':
    main()
