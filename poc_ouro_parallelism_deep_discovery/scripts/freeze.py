"""Select metadata-only cohorts and falsification criteria before model outcomes."""
import datetime
import random
from poc_ouro_parallelism_deep_discovery.contract import *

def main():
    target=ROOT/'artifacts/probe_protocol.json'
    assert not target.exists(),'Never replace a frozen protocol.'
    full=read(PROJECT/'poc_decode_first_depth_aware_dp/artifacts/quality_protocol_v3_pilot32.json')
    previous=read(PROJECT/'poc_depth_cdb_factorial/artifacts/workload_frozen.json')
    excluded={r['id'] for r in previous['quality']}|{r['id'] for r in previous['speed']}
    excluded.update(r['id'] for r in full['rows'] if r['split']=='pilot')
    chosen=[]
    for domain in ['gsm8k','math500']:
        candidates=[r for r in full['rows'] if r['dataset']==domain and r['id'] not in excluded]
        random.Random(2026101907).shuffle(candidates)
        for i,row in enumerate(candidates[:40]):
            chosen.append(dict(row,campaign_partition='calibration' if i<8 else 'held_out'))
    assert len(chosen)==80
    write(target,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=chosen,
        selection_seed=2026101907,selection='Metadata-only shuffle, excluding the preceding factorial and pilot IDs.',
        calibration_per_domain=8,held_out_per_domain=32,
        source=full['source'],source_sha256=sha(PROJECT/'poc_decode_first_depth_aware_dp/artifacts/quality_protocol_v3_pilot32.json'),
        checkpoint_pins={k:dict(id=v['id'],revision=v['revision'],layers=v['layers']) for k,v in MODELS.items()},
        BF16=True,temperature=0,EOS=0,quality_cap=8192,
        fixed=dict(min_R=4,max_R=4,q=1.0,KV='depth_indexed full R4'),
        adaptive=dict(min_R=2,max_R=4,q=.5,KV='last_exited',gate='original checkpoint gate',prefill='full R4'),
        component_shapes=dict(rows=[1,2,4,8,16,32,128,512],contexts=[1024,4096,8192],P=[2,4]),
        component_scope='Actual pretrained weights. Captured calibration activations where available. Not E2E speedup.',
        stop_criteria={
          'remote_operator_pool':'Prune a layout if round-trip is slower than a matched native operator even with ideal ready work.',
          'parallel_time':'Prune a full-Jacobian formulation if its minimum charged work defeats finite-R4 critical-path saving.',
          'residual_projection':'Count all fresh residual GEMMs. No sparse or low-rank assumption without measurement.',
          'fixed_KV_alias':'Reject cross-depth aliasing if actual cached K/V differ. Do not substitute a shared-KV model.',
        },
        quality_margins_pp=dict(gsm8k=1.,math500=2.),
        quality_status='No candidate quality claim. Pilot detects collapse, not narrow noninferiority.',
        P8='NOT_RUN. Projection only.',training='NONE'))
    print('FROZEN',len(chosen),'questions. No candidate outcomes used.')

if __name__=='__main__': main()
