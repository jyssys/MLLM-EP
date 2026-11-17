"""Match actual cut counts and FIFO geometry, not only planned budgets."""
import collections
import fcntl
import json
import random
import time
from .common import ROOT,QUIET_LOCK,resources,save,write
from .run import execute
from .analysis import metrics,records
from .heldout_baselines import SEEDS


def control_config(name,source,seed):
    cfg=json.loads((ROOT/'configs'/(source+'.json')).read_text())
    commits=records(ROOT/'raw'/source/'actual_cut_commits.jsonl')
    groups=[r['requests'] for r in commits]
    assert [rid for group in groups for rid in group]==['q'+str(i) for i in range(cfg['count'])]
    cuts=[r['offloaded_layers'] for r in commits]
    bysize=collections.defaultdict(list)
    for i,group in enumerate(groups):
        bysize[len(group)].append(i)
    rng=random.Random(seed+991)
    shuffled=list(cuts)
    for indices in bysize.values():
        values=[cuts[i] for i in indices];rng.shuffle(values)
        for index,value in zip(indices,values):
            shuffled[index]=value
    actual=[m for group,m in zip(groups,cuts) for _ in group]
    matched=[m for group,m in zip(groups,shuffled) for _ in group]
    assert collections.Counter(actual)==collections.Counter(matched)
    cfg.update(name=name,policy='STATIC',evaluation_arm='MatchedHistogramControl',
        confirmation_method='MatchedHistogramControl',flow_balance_policy=False,
        preplanned_source_groups=True,planned_source_group_sizes=[len(g) for g in groups],
        static_layers_by_request=matched,static_fraction_axis='REQUEST_ID_BLOCKS',
        phase='POSTHOC_MATCHED_AGGREGATE_AND_GEOMETRY_CONTROL',
        posthoc_aggregate_information=True,control_source=source,
        control_scope='Use only aggregate cut counts within source-group sizes and recorded FIFO group membership. Randomize cut placement without reading request latency, deadlines or future completion. This is a retrospective matched control, not a deployable static policy selected before the held-out trace.')
    return cfg


def compare_geometry(reference,candidate):
    def normalized(groups):
        return [sorted(group,key=lambda rid:int(rid[1:])) for group in groups]
    a,b=metrics(reference),metrics(candidate)
    fields=('completed_migrated_requests','completed_layer_work','D_GEMM_token_layer_rows',
        'valid_KV_payload_bytes','hidden_residual_payload_bytes')
    checks=dict(cut_histogram=a['actual_cut_histogram']==b['actual_cut_histogram'],
        source_groups=normalized(a['source_groups'])==normalized(b['source_groups']),
        destination_groups=normalized(a['destination_groups'])==normalized(b['destination_groups']),
        work=all(a['work_accounting'][key]==b['work_accounting'][key] for key in fields))
    result=dict(reference=reference,candidate=candidate,checks=checks,passed=all(checks.values()),
        evidence='TRACE_ANALYSIS',scope='Compare actual completed work and actual group membership. Planned matching alone is insufficient. A failed budget or geometry check cannot support an independent policy-gain claim.')
    write(ROOT/'analysis'/(candidate+'_actual_matching_gate.json'),result)
    return result


def main():
    values=[]
    try:
        for seed in SEEDS:
            source=f'heldout_Dynamic_N128_O16_q6.4_s{seed}_v1'
            path=ROOT/'raw'/source/'dispatcher_result.json'
            heartbeat=0.
            while not path.exists() or not json.loads(path.read_text()).get('complete'):
                if time.time()>json.loads((ROOT/'state/progress.json').read_text())['deadline_epoch']-4800:
                    write(ROOT/'state/not_run_matched_policy_controls.json',dict(status='NOT_RUN',reason='Preserve closeout time',pending_source=source))
                    return
                if time.time()-heartbeat>=60:
                    heartbeat=time.time()
                    print('WAIT_MATCHED_CONTROL_SOURCE',source,flush=True)
                time.sleep(15)
            name=f'heldout_histogram_geometry_control_N128_O16_q6.4_s{seed}_v1'
            cfg=control_config(name,source,seed)
            if not execute(save(cfg)):
                continue
            gate=compare_geometry(source,name)
            values.append(dict(source=source,name=name,gate=gate,metrics=metrics(name)))
            write(ROOT/'analysis/heldout_exact_histogram_geometry_controls.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    import sys
    if '--conditional64' in sys.argv:
        from .conditional_control import main as conditional_main
        conditional_main()
    else:
        main()
