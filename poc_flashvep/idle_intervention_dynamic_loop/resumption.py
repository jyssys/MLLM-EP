"""Resume only named unfinished gates. Keep earlier receipts immutable."""
import copy
from .common import *


def main():
    assert RESUMPTION is not None, 'An explicit bounded resumption record is required'
    initialize()
    snapshot=ROOT/'state/pre_resumption_20261109.json'
    if not snapshot.exists():
        files=('budget.json','active.json','queue_owner.json','monitor_snapshot.json',
            'monitor_finished.json','deadline_guard.json','queue.json')
        write(snapshot,{name:json.loads((ROOT/'state'/name).read_text())
            for name in files if (ROOT/'state'/name).exists()})
    jobs=[]
    for model in ('ouro','huginn'):
        for variant in ('static','dynamic'):
            old=f'n8_{model}_pinned_wire_{variant}_state8'
            cfg=copy.deepcopy(json.loads((ROOT/'configs'/(old+'.json')).read_text()))
            cfg['name']=f'r1_{model}_pinned_{variant}_state8'
            if model=='huginn':cfg['D_kv_blocks']=23552
            cfg.update(timeout=600,startup_timeout=300,resumption_record='state/resumption_20261109.json')
            save(cfg);jobs.append(cfg['name'])
    # Recheck the clean N32 variable-count geometry with full-byte observers.
    # Observer controls do not establish clean Decode protection.
    for model in ('ouro','huginn'):
        source=(f'pred_{model}_tile_relative_clean32'+('_workspace_v2' if model=='huginn' else ''))
        cfg=copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
        cfg.update(name=f'r1_{model}_actual_mixed_state32',state_probe=True,verify_transfer=False,
            async_prompt_state_probe=True,async_verify_handoff=True,
            deferred_prompt_state_probe=False,qualification_ignore_observer_tail_latch=True,
            adaptive_decode_reserve=False,timeout=900,startup_timeout=300,
            phase='ACTUAL_BOUNDARY_FULL_BYTE_STATE_DIAGNOSTIC_NOT_CLEAN_PERFORMANCE',
            resumption_record='state/resumption_20261109.json',
            observer_scope='Retain fixed independently calibrated reserves during full-byte state copying. Native-relative performance is not assessed in this diagnostic.')
        save(cfg);jobs.append(cfg['name'])
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text())
    write(path,queue+[name for name in jobs if name not in queue])
    print(json.dumps(dict(added=jobs,deadline=RESUMPTION),indent=2))


if __name__=='__main__':main()
