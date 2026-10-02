"""CPU-only compact export. Never publish hidden tensors or model weights."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from poc_dispatch_combine_asymmetry.contract import ROOT, write
from poc_dispatch_combine_asymmetry.scripts.analyze import native


def main():
    accounting=ROOT/'artifacts/communication_accounting.json'
    with gzip.GzipFile(filename=str(accounting)+'.gz',mode='wb',mtime=0) as stream:
        stream.write(accounting.read_bytes())
    arrays={};index=[]
    for tag in ('llm_controls_v3','mllm_internal_v3'):
        for path in sorted((ROOT/f'artifacts/runs/{tag}/captures').glob('*.pt')):
            x=torch.load(path,map_location='cpu',weights_only=False)
            key=f'r{len(index)}';record=x['record']
            arrays[key+'_ids']=x['ids'].numpy()
            arrays[key+'_weights']=x['weights'].float().numpy()
            arrays[key+'_owner']=x['expert_map'].numpy()
            index.append({'key':key,'source_run':tag,**{k:record[k] for k in ('model','arm','size','wave','layer','ep_rank','prompt_tokens')},
                'hidden_published':False,'weights_scope':'router scalar weights only, NOT checkpoint expert weights'})
    assert len(index)==9*48*4
    np.savez_compressed(ROOT/'artifacts/compact_routes.npz',**arrays)
    write(ROOT/'artifacts/compact_routes_index.json',index)
    repo=ROOT.parent
    workload=repo/'poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json'
    original=json.loads(workload.read_text())
    write(ROOT/'artifacts/frozen_workload.json',{'source':str(workload),'sha256':hashlib.sha256(workload.read_bytes()).hexdigest(),
        'q2_visual_sweep':original['q2_visual_sweep'],'text_template':'reused matched_text; actual tokens 174/724/2033',
        'selection':'frozen prior requests, no routing/time based sample selection; single input per arm-size'})
    mapping=[]
    for file in (ROOT/'artifacts/runs/mllm_clean_v2').glob('stage.pid*.jsonl'):
        row=json.loads(file.read_text().splitlines()[0]);device=row['logical_cuda_device']
        physical=int(row['worker_cuda_visible_devices'].split(',')[device])
        assert physical==[0,1,4,5][row['ep_rank']]
        mapping.append({'physical_gpu':physical,'logical_cuda_device':device,'ep_rank':row['ep_rank'],
            'tp_rank':row['tp_rank'],'dp_rank':row['dp_rank'],'pid':int(file.stem.removeprefix('stage.pid')),
            'worker_cuda_visible_devices':row['worker_cuda_visible_devices'],'expert_id_range':[row['ep_rank']*32,row['ep_rank']*32+31],
            'expert_weight_bytes_each':row['expert_weight_bytes_each']})
    write(ROOT/'artifacts/runtime_rank_mapping.json',sorted(mapping,key=lambda r:r['ep_rank']))
    paths=[*ROOT.glob('*.py'),*ROOT.glob('scripts/*.py'),*ROOT.glob('hooks/*.py'),
        repo/'poc_native_lp_ep4/scripts/gpu_session.py',repo/'poc_native_lp_ep4/scripts/network.py',
        repo/'poc_mllm_ep_communication_motivation/scripts/run_q1_q2.py',
        repo/'poc_mllm_ep_communication_motivation/scripts/run_q3_interference.py',
        repo/'poc_mllm_ep_foundation/artifacts/frozen_baseline_bundle.json',
        Path('/home/esjung/.cache/flashvep-deepep-v020/DeepEP/csrc/deep_ep.cpp'),
        Path('/home/esjung/.cache/flashvep-deepep-v020/DeepEP/csrc/kernels/intranode.cu')]
    write(ROOT/'artifacts/source_integrity.json',{'files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        'scope':'final published code/source and reused bundle; not a retroactive per-run source snapshot',
        'code_history':'primary v2 stage hook preceded optional profiler/envelope additions; no per-run full untracked source snapshot was recorded; same prepare/expert/finalize boundaries retained'})
    kernel=json.loads((ROOT/'artifacts/nsight/kernel_classification.json').read_text())
    profile=[]
    for tag in ('mllm_nsight_v4','llm_nsight_v4'):
        run=ROOT/f'artifacts/runs/{tag}'
        pids={json.loads(f.read_text().splitlines()[0])['ep_rank']:int(f.stem.removeprefix('stage.pid')) for f in run.glob('stage.pid*.jsonl')}
        for s in native(run):
            pid=pids[s['critical_rank']]
            selected=next(r for r in kernel['summaries'] if r['os_pid']==pid and r['arm']==s['arm'])
            profile.append({'model':s['model'],'arm':s['arm'],'critical_rank':s['critical_rank'],'os_pid':pid,
                'classes':selected['classes'],'stage_intervals_ms':{k:s[k] for k in ('dispatch_ms','expert_ms','combine_ms')},
                'label':'KERNEL_CLASSIFIED','scope':'one profiler repeat, same cumulative-critical rank; NOT clean baseline; fused boundaries not exclusive decomposition'})
    write(ROOT/'artifacts/profile_critical_rank_classification.json',profile)
    table='| Arm | selected EP rank | layout kernel | count/barrier notify | fused dispatch | cached combine notify | fused combine |\n|---|---:|---:|---:|---:|---:|---:|\n'
    names=('dispatch_layout','dispatch_notify_counts_barriers','dispatch_pack_move_receive_poll',
           'combine_cached_notify_barriers','combine_reverse_move_reduce_poll')
    for row in profile:
        table+=f"| {row['model']} {row['arm']} | {row['critical_rank']} | "+' | '.join(f"{row['classes'][n]['mean_ms']:.4f}" for n in names)+' |\n'
    (ROOT/'reports/kernel_classification.md').write_text('''# Physical kernel boundary evidence (diagnostic only)

All entries below are **KERNEL_CLASSIFIED**, ms/call, one separate Nsight repeat per arm, mean of48 calls on ONE cumulative-critical rank selected within that profiled run. These are NOT clean baseline decomposition fractions. Each same-rank classifier matches actual OS PID to CUPTI launch correlation. Twelve native rank/arm ranges each contain exactly48 instances of all five classes.

'''+table+'''
`notify_dispatch` combines count exchange/prefix computation/barriers; it is not a pure count arithmetic or wire interval. Dispatch fuses packing, copying, receive polling; combine fuses reverse copy, reduction and polling. Kernel durations may contain rendezvous waits. Host polling/allocation/launch gaps and vLLM remapping are not separately measured here. No cross-rank timestamp subtraction or sum of unrelated maxima is used.

The count/rendezvous kernel is much larger than layout arithmetic and fused payload kernels on the selected ranks in this diagnostic. This supports a count/rendezvous/tail explanation but cannot assign an exclusive percentage of the CLEAN dispatch cost. Nsight changes producer readiness; the profiled first-layer duration is not interchangeable with the clean~29.60ms waiting-rank interval. Only source-backed classification and own-lane duration are claimed.

Full per-rank/per-call records: `artifacts/nsight/kernel_classification.json`; same-rank selected summaries: `artifacts/profile_critical_rank_classification.json`.
''')
    ledger=[]
    for run in sorted((ROOT/'artifacts/runs').iterdir()):
        worker=run/'worker.dp0.json'
        if not worker.exists():continue
        records=json.loads(worker.read_text())['records']
        ledger.append({'run':run.name,'real_requests_including_warmup':len(records),
            'strata':dict(collections.Counter(f"{r['model']}_{r['arm']}_{r['size']}_{r['lane']}" for r in records)),
            'excluded_from_headline':run.name not in ('mllm_clean_v2','llm_controls_v3'),
            'status':json.loads((run/'status.json').read_text())})
    write(ROOT/'artifacts/execution_ledger.json',{'runs':ledger,'total_real_requests_including_excluded_attempts':sum(r['real_requests_including_warmup'] for r in ledger),
        'idle_DP_calls':'not extra user requests, separately preserved in stage/worker logs',
        'replay':'3 captured layers × 3 scales × 2 arms × (3 warmup+10 repeat), no model inference',
        'quality_evaluation_repeated':False,'optimization_implemented':False})
    print('PUBLISHED_COMPACT_ROUTE_TENSORS',len(index))


if __name__=='__main__':main()
