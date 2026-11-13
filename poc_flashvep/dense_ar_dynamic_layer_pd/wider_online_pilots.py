"""Test wider, qualified destination cohorts without changing the SLO."""
import fcntl
from .common import *
from .online_pilots import serving_config
from .run import execute
from .analyze import compare,metrics,records


def main():
    initialize()
    gate=json.loads((ROOT/'state/D8_mixed_protocol_qualified.json').read_text())
    assert gate['passed'] and gate['actual_offloaded_layer_counts']==[0,2,4,8,12]
    results=[]
    try:
        for cap in (16,32,64):
            for mode in ('COST','DEADLINE','IDLE'):
                c=serving_config(f'online_D8_{mode}_O{cap}_N64_v1',cap=cap,policy=mode,
                    D_token_budget=64000,phase='WIDER_QUALIFIED_DESTINATION_ONLINE_DISCOVERY')
                if not execute(save(c),restore=False):return
                output=compare(f'coarse_O{cap}_N64_q4.777_native16_s1_v1',c['name'])
                commits=records(ROOT/'raw'/c['name']/'actual_cut_commits.jsonl')
                distribution={str(n):sum(len(r['requests']) for r in commits if r['offloaded_layers']==n)
                    for n in (0,2,4,8,12)}
                row=dict(name=c['name'],output_gate=output,metrics=metrics(c['name']),
                    actual_layer_distribution=distribution)
                results.append(row);write(ROOT/'state/wider_online_pilot_progress.json',dict(results=results))
                print('WIDER_ONLINE',c['name'],output['passed'],distribution,row['metrics']['TTFT_p95_ms'],
                    row['metrics']['ITL_p99_ms'],flush=True)
                if not output['passed']:return
        write(ROOT/'analysis/wider_online_pilot_results.json',dict(results=results,
            scope='Discovery only. Batch8 suffixes fit the real64000-token destination budget. Costs above batch4 use a conservative extrapolation.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
