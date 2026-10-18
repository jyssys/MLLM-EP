"""Equal-progress accounting and native refill diagnostics."""
from poc_decode_first_depth_aware_dp_v3.contract import *
import statistics

def main():
    frozen=read(ROOT/'artifacts/discovery_frozen.json')['rows']
    windows=[]
    for k in [32,64,128]:
        values=[]
        for r in frozen:
            raw=read(ROOT/'outputs/discovery128_graph_b1_v1'/(r['id'].replace('/','_')+'.json'))
            if len(raw['depth_events'])<k:continue
            values.append(dict(id=r['id'],dataset=r['dataset'],K=k,
                mean_depth=statistics.mean(e['depth'] for e in raw['depth_events'][:k])))
        windows.append(dict(K=k,n=len(values),mean_depth=statistics.mean(v['mean_depth'] for v in values),
            between_request_std=statistics.pstdev(v['mean_depth'] for v in values),rows=values,
            scope='Naturally reached decode-token windows. No forced continuation. No standalone window-latency claim.'))
    write(ROOT/'artifacts/equal_progress_windows.json',windows)
    ablations=[]
    for mode in ['fixed','refill','no_refill']:
        folder=ROOT/f'outputs/ablation_{mode}_v1'
        assert read(folder/'status.json')['success']
        batches=[read(p) for p in folder.glob('batch_*.json')]
        ablations.append(dict(mode=mode,requests=8,local_batch=4,cap=64,
            wall_ms_sum=sum(b['wall_ms'] for b in batches),
            executed_recurrent_work=sum(b['stats']['recurrent_steps'] for b in batches),
            recurrent_batches=sum(b['stats']['recurrent_batches'] for b in batches),
            mean_active_recurrent_batch=sum(b['stats']['recurrent_steps'] for b in batches)/sum(b['stats']['recurrent_batches'] for b in batches),
            refill_events=sum(b['stats']['scheduler_refills'] for b in batches),
            scope='Physical calibration diagnostic, not a matched-quality speedup claim.'))
    write(ROOT/'artifacts/ablation_summary.json',ablations)

if __name__=='__main__':main()
