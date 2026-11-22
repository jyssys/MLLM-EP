"""Measure matched continuous paths and keep normal compiled baselines visible."""
import json
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .next_jobs import plan


def main():
    capsule,sha=freeze('continuous_static_discovery_v1_20261121')
    data=json.loads((ROOT/'data/balanced400_discovery.json').read_text())
    suffixes,work=matched_suffixes([r['input_tokens'] for r in data],4)
    lanes=[[],[]];cells=[]
    for q_index,qps in enumerate((1.8,2.8)):
        first=(2,6)[q_index];jobs=[]
        for label,axis,compile_mode,credit in (
                ('NativeMatched','Native','MATCHED_NO_COMPILE',4),
                ('NativeDefault','Native','INSTALLED_DEFAULT_COMPILE',0),
                ('AggDefault','Agg','INSTALLED_DEFAULT_COMPILE',0),
                ('Layer4','Layer','MATCHED_NO_COMPILE',4),
                ('Token4','Token','MATCHED_NO_COMPILE',4),
                ('Hybrid4_f0125','Hybrid','MATCHED_NO_COMPILE',4)):
            extra={}
            if axis in ('Layer','Token','Hybrid'):extra=dict(qualification_gate='C3_continuous_actions_gate_v1.json')
            if axis=='Token':extra.update(suffix_by_request={'q'+str(i):s for i,s in enumerate(suffixes)},aggregate_work_audit=work)
            if axis=='Hybrid':extra.update(hybrid_suffix_fraction=.125,partition_aligned_chunks=True)
            if axis!='Agg':extra['continuous_source_credit_limit']=credit
            name=create(f'C3Static_Discovery200_q{qps}_{label}_v1',axis,'balanced400_discovery.json',
                cut=4 if axis in ('Layer','Token','Hybrid') else 0,pair=(first,first+1),qps=qps,seed=2026112291,
                execution_family=compile_mode,source_capsule=capsule,source_capsule_SHA256=sha,
                discovery_only=True,continuous_static_discovery=True,
                comparison_scope='Matched mode0/source-credit4 paths are separate from default-compiled uncapped Native/Agg. '
                    'This initial cut4 comparison is not the final best-tuned static comparison. '
                    'Each method uses its actual continuous batches. Natural outputs may differ under the declared BI0 quality contract.',**extra)
            jobs.append(name)
        lanes[q_index].extend(jobs);cells.append(dict(QPS=qps,jobs=jobs))
    write(ROOT/'configs/continuous_static_initial_discovery_v1.json',dict(cells=cells,
        source_capsule=capsule,source_capsule_SHA256=sha,
        discovery_requests=len(data),heldout_requests_used_for_tuning=0,
        scope='Initial same-arrival continuous measurement. No best-static or serving-success claim follows from one start.'))
    for lane,jobs in enumerate(lanes):plan(f'continuous_static_initial_lane{lane}_v1',jobs)
    print('Frozen12 initial continuous discovery starts.',flush=True)


if __name__=='__main__':main()
