"""Compare generic native options without calling them a new architecture."""
import argparse
import statistics
import numpy as np
from poc_ouro_parallelism_deep_discovery.contract import *
from poc_ouro_parallelism_deep_discovery.scripts.native_parity_analysis import stats,log_softmax

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists()
    baselines={};records=[]
    for root in sorted((ROOT/'outputs').glob('native_base*')):
        for state in root.glob('*_R*_TP*/status.json'):
            s=read(state)
            if s.get('success'):baselines[s['size'],s['R'],s['TP']]=state.parent
    for root in sorted((ROOT/'outputs').glob('control_*')):
        for state in root.glob('*_R*_TP*/status.json'):
            s=read(state)
            if not s.get('success'):continue
            key=s['size'],s['R'],s['TP']
            if key not in baselines:continue
            base=baselines[key];config=read(state.parent/'engine_config.json')
            samples=[read(f) for f in state.parent.glob('P*_request*_repeat*.json')]
            native=[read(f) for f in base.glob('P*_request*_repeat*.json')]
            rows=[]
            for N in [1024,4096,8192]:
                current=[v for v in samples if v['prompt']==N];reference=[v for v in native if v['prompt']==N]
                assert len(current)==len(reference)==10
                left=statistics.median(v['wall_ms'] for v in reference);right=statistics.median(v['wall_ms'] for v in current)
                rows.append(dict(context=N,default_median_E2E_ms=left,control_median_E2E_ms=right,
                    default_over_control_speedup=left/right,N_trials_each=10,N_distinct_shapes=2,
                    uncertainty='Sequential method order. Two request shapes, not broad request-level confirmation.'))
            source=read(base/'parity_capture.json')['evidence'][0]
            target=read(state.parent/'parity_capture.json')['evidence'][0]
            a,b=np.load(source['path']),np.load(target['path'])
            assert source['KV_names']==target['KV_names'] and source['positions']==target['positions']
            lp,lq=log_softmax(a['logits']),log_softmax(b['logits'])
            records.append(dict(model=key[0],R=key[1],P=key[2],run=root.name,default_run=base.parent.name,
                async_scheduling=config['async_scheduling'],
                fuse_allreduce_rms=config['compilation_config'].get('pass_config',{}).get('fuse_allreduce_rms',False),
                controlled_work=rows,
                numerical=dict(logits=stats(b['logits'],a['logits']),hidden=stats(b['hidden'],a['hidden']),KV=stats(b['KV'],a['KV']),
                    top1_match=(a['logits'].argmax(-1)==b['logits'].argmax(-1)).tolist(),
                    KL=(np.exp(lp)*(lp-lq)).sum(-1).tolist()),
                quality='Four-position controlled capture only. Natural option-on/off accuracy and full-generation parity NOT_RUN.',
                evidence_class='LIVE_E2E',label='Generic native baseline control. Not candidate architecture performance.'))
    write(args.output,dict(records=records,
        interpretation='An existing native option can invalidate apparent headroom over a weaker configuration. Keep the stronger validated option in future same-P comparisons.',
        candidate='NONE',natural_candidate_quality='NOT_RUN'))
    for r in records:print(r['run'],[(x['context'],round(x['default_over_control_speedup'],4)) for x in r['controlled_work']],r['numerical']['logits']['bitwise_float_capture_equal'])

if __name__=='__main__':main()
