"""Check paired inputs and unchanged loaded runtime code across confirmation."""
from collections import defaultdict
from .common import *


def main():
    plan=json.loads((ROOT/'heldout_plan.json').read_text())
    groups=defaultdict(list);hashes=defaultdict(set);completed=[];missing=[]
    keys=('model','model_revision','depth','length','cap','count','gpus','P_slots','D_slots',
        'kv_blocks','D_kv_blocks','deep_graph_budget','P_prefill_graph_shapes','dataset','prompt_ids',
        'seed','request_caps','arrival_seed','arrival_offsets','native_ITL_target_ms',
        'initial_decode_reserve_by_batch','decode_guard_ms','guard_ms','admission_horizon_ms',
        'empty_decode_budget_ms','empty_quantum_cap_ms','empty_admission_uses_current_cap')
    core=('runtime.py','protocol.py','warmup.py','bootstrap/sitecustomize.py')
    for row in plan['jobs']:
        name=row['name'];cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        assert digest(ROOT/'configs'/(name+'.json'))==row['configuration_SHA256']
        groups[cfg['model'],cfg['arrival_seed']].append(cfg)
        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
            missing.append(name);continue
        completed.append(name)
        manifests=list(folder.glob('loaded_intervention_source_*.json'))
        assert manifests
        for file in manifests:
            source=json.loads(file.read_text())['hashes']
            for suffix in core:
                matches=[sha for path,sha in source.items() if path.endswith('/idle_intervention_dynamic_loop/'+suffix)]
                assert len(matches)==1,(name,suffix)
                hashes[suffix].add(matches[0])
    for cfgs in groups.values():
        reference={key:cfgs[0][key] for key in keys}
        assert all({key:cfg[key] for key in keys}==reference for cfg in cfgs)
    assert all(len(values)==1 for values in hashes.values())
    result=dict(time=time.time(),paired_configuration_contract_passed=True,
        loaded_core_source_consistent=True,loaded_core_SHA256={k:next(iter(v)) for k,v in hashes.items()},
        completed_starts=completed,incomplete_starts=missing,
        checked_configuration_fields=keys,
        scope='Configuration hashes match the frozen plan. Shared inputs, resource settings, and estimator fields match within each paired seed. Analysis helpers can change without changing loaded runtime code. Missing starts remain explicit.')
    write(ROOT/'confirmation_contract.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
