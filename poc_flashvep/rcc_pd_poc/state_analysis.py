"""Compare valid prompt KV planes and final prompt hidden states across arms."""
import json
from .common import ROOT, write


def main():
    runs = {}
    for folder in sorted((ROOT/'raw').glob('state128*')):
        if not (folder/'window.json').exists():
            continue
        window = json.loads((folder/'window.json').read_text())
        records = [r for p in folder.glob('prompt_state_probe_*.json')
                   for r in json.loads(p.read_text())['records'] if r['host_time'] >= window['start']]
        runs[folder.name] = {r['seed']:r for r in records}
    result = []
    for name, actual in runs.items():
        cfg = json.loads((ROOT/'raw'/name/'rcc_contract.json').read_text())
        for arm in ('PD_G1','A_sep'):
            reference_name = name.replace('_'+cfg['arm']+'_','_'+arm+'_')
            if reference_name == name or reference_name not in runs:
                continue
            expected = runs[reference_name]
            for seed, record in actual.items():
                other = expected.get(seed)
                if other is None:
                    raise RuntimeError('The reference prompt state is missing')
                mismatches = [dict(depth=a['depth'],layer=a['layer'],kind=a['kind'])
                              for a,b in zip(record['planes'],other['planes']) if a != b]
                result.append(dict(run=name, reference=reference_name, seed=seed,
                                   prompt_tokens=record['prompt_tokens'], hidden_exact=record['hidden_sha256']==other['hidden_sha256'],
                                   plane_count=len(record['planes']), KV_planes_exact=record['planes']==other['planes'],
                                   divergent_planes=mismatches,
                                   evidence='MEASURED_HEAVY_CORRECTNESS'))
    write(ROOT/'prompt_state_comparisons.json', result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
