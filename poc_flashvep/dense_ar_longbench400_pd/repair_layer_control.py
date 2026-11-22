"""Keep the frozen Layer baseline in the independent decode-repair comparison."""
import argparse
import json
import time
from .common import ROOT,write,digest,dispatch_allowed
from .final_confirmation_jobs import clone
from .run import save,execute
from .quality_gate import compare as quality
from .qualification import audit
from .metrics import summarize


PLAN='hybrid_READ_repair_layer_control_extension_v1.json'


def build():
    assert not (ROOT/'configs'/PLAN).exists()
    repair_path=ROOT/'configs/hybrid_READ_repair_frozen_v1.json'
    repair=json.loads(repair_path.read_text())
    blocks=[c for c in repair['cells'] if c['stage']=='INDEPENDENT_REPAIR_CONFIRMATION']
    assert all(not (ROOT/'raw'/name).exists() for c in blocks for name in c['jobs'].values()),'Freeze this baseline before repair confirmation timings'
    main=json.loads((ROOT/'configs/final_continuous_execution_plan_v1.json').read_text())
    original=next(c for c in main['cells'] if c['workload']=='Main400' and c['QPS']==1.8 and c['start']==2)
    prototype=json.loads((ROOT/'configs'/(original['jobs']['BestLayer']+'.json')).read_text())
    cells=[]
    for c in blocks:
        name=f'RepairLayerConfirm_Main400_q1.8_s{c["start"]}_v1'
        cfg=clone(prototype,name,c['dataset'],c['QPS'],c['seed'],c['GPU_pair'],
            repair['source_capsule'],repair['source_capsule_SHA256'])
        cfg.update(final_composition_gate=prototype['final_composition_gate'],repair_layer_control=True,
            scope='The original discovery-frozen Layer action is an added baseline,not a new tuning choice. '
                'Same new repair seeds,requests,source,pair and natural output contract. '
                'Runs after the original seven-arm repair blocks,not in their randomized dispatch order. '
                'Preserve this time-block distinction and all failed outcomes.')
        save(cfg);cells.append(dict(stage='ADDED_FROZEN_LAYER_CONTROL',QPS=c['QPS'],start=c['start'],
            seed=c['seed'],GPU_pair=c['GPU_pair'],dataset=c['dataset'],jobs=dict(LayerControl=name)))
    write(ROOT/'configs'/PLAN,dict(time=time.time(),cells=cells,
        source_capsule_SHA256=repair['source_capsule_SHA256'],repair_design_SHA256=digest(repair_path),
        prototype=original['jobs']['BestLayer'],evidence='PRECONFIRMATION_ADDITIONAL_BASELINE_DESIGN',
        scope='Three extra starts add the already-frozen Layer to Native/Token/Hybrid decode repair. '
            'Do not change the original140 main starts or22 repair starts. '
            'No repaired confirmation timing was observed when this extension froze. '
            'These Layer starts follow the seven-arm repair blocks. They are not jointly randomized with them.'))
    print('ADDED_FROZEN_LAYER_REPAIR_CONTROL',3,'starts; no earlier design replaced',flush=True)


def run():
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    while not (ROOT/'state/hybrid_READ_repair_complete_v1.json').exists():
        assert dispatch_allowed();time.sleep(10)
    results=[]
    for c in plan['cells']:
        name=c['jobs']['LayerControl'];success=execute(name)
        if success:
            state=audit(name);assert state['passed'] and state['decode_full36_observation_complete']
            score=quality(name);m=summarize(name)
            result=dict(name=name,status='MEASURED_ONLINE' if score['passed'] else 'FAILED_TASK_QUALITY',
                quality=score,ITL_p99_ms=m['ITL_p99_ms'],TPOT_p95_ms=m['TPOT_p95_ms'],TTFT_p95_ms=m['TTFT_p95_ms'])
        else:result=dict(name=name,status='FAILED')
        results.append(result)
        write(ROOT/'state/hybrid_READ_repair_layer_control_progress_v1.json',dict(results=results))
    write(ROOT/'state/hybrid_READ_repair_layer_control_complete_v1.json',dict(complete=True,results=results,
        scope='Terminal execution does not imply decode or serving success. Keep the separate time block visible.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('build','run'));args=parser.parse_args()
    build() if args.action=='build' else run()
