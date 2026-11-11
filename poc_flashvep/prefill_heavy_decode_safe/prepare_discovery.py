"""Prepare disjoint capacity data and a bounded workload discovery sequence."""
import copy,json,shutil
from .common import *


def main():
    initialize();source=REPO/'poc_flashvep/reports/looped_pd_autonomous_prefill_compute_decode_headroom'
    datasets=[];jobs=[]
    for model,lengths in (('ouro',(4096,8192)),('huginn',(2048,3072,3500))):
        anchor=ROOT/'configs'/f'anchor_{model}_L{4096 if model=="ouro" else 2048}_O{32 if model=="ouro" else 16}_clean_cupti26_N64_a2026111210.json'
        template=json.loads(anchor.read_text())
        for length in lengths:
            for split in ('calibration','confirmation'):
                filename=f'{split}_{model}_L{length}.json';original=source/'data'/filename
                target=ROOT/'data'/filename
                if target.exists():assert target.read_bytes()==original.read_bytes()
                else:shutil.copyfile(original,target)
                rows=json.loads(target.read_text());ids=[r['prompt_id'] for r in rows]
                assert len(ids)==len(set(ids)),('Repeated source IDs',filename)
                assert all(len(r['input_ids'][model])==length for r in rows)
                assert length+template['cap']<=template['context_limit']
                datasets.append(dict(file=filename,SHA256=digest(target),count=len(rows),distinct_prompt_ids=len(set(ids)),
                    original=str(original),purpose='Independent capacity calibration' if split=='calibration' else 'Discovery cohort, not final confirmation'))
            cfg=copy.deepcopy(template);pool=json.loads((ROOT/'data'/f'calibration_{model}_L{length}.json').read_text())
            cfg.update(name=f'capacity_{model}_L{length}_B124_start1',worker='capacity_worker',
                phase='INDEPENDENT_FINITE_CAPACITY_REFERENCE',length=length,state_capacity_tokens=length,
                chunk=length,token_budget=4*length,capacity=4,slots=4,P_slots=4,graph_rows=4,
                P_prefill_graph_rows=4*length,P_prefill_graph_shapes=64,D_prefill_graph_rows=length,
                kv_blocks=24832 if model=='huginn' else 16384,P_graph_local_capacity=True,capacity_mode='P_AND_PREPARED_D',
                jobs=[dict(batch=b,cap=template['cap'],repeats=3) for b in (1,2,4)],
                dataset=f'calibration_{model}_L{length}.json',count=4,prompt_ids=[r['prompt_id'] for r in pool[:4]],
                arrival_offsets=[0.]*4,request_caps=[template['cap']]*4,discovery_activity=False,
                state_probe=False,verify_transfer=False,timeout=900,startup_timeout=300,
                data_scope=dict(distinct_prompt_ids=4,pool_records=len(pool),prefix_caching=False,cyclic_reuse=False,
                    scope='Independent finite local P and prepared valid-state D cohorts. No sustainable capacity claim.'))
            save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/discovery_datasets.json',datasets)
    write(ROOT/'state/capacity_discovery_plan.json',dict(jobs=jobs,
        scope='Calibration precedes adaptive QPS selection. B4 that exceeds legal KV credit is recorded as blocked, not silently removed.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
