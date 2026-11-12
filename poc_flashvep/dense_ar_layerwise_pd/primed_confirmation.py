"""Confirm the exact text-enabled family without an inflated Native tail."""
import json, random, re
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics, compare
from .graph_feasibility import gate
from .resume_campaign import restore, record


def main():
    prior=json.loads((ROOT/'state/primed_campaign_progress_v31.json').read_text())['results']
    assert all(r['passed'] for r in prior if 'passed' in r)
    native=[r for r in prior if 'metrics' in r and '_native' in r['name']]
    assert len(native)==2 and all(r['metrics']['Decode_absolute_SLO_pass'] for r in native)
    results=[]
    try:
        for cut in (35,32):
            for count in (8,32):
                c=json.loads((ROOT/'configs'/f'primed_graph_qwen8_L8000_O16_N{count}_c28_qual_v31.json').read_text())
                c.update(name=f'primed_graph_qwen8_L8000_O16_N{count}_c{cut}_qual_v32',layer_cut=cut)
                name=save(c)
                if not execute(name,restore=False):return
                passed=gate(name,count,True)
                results.append(dict(name=name,passed=passed))
                write(ROOT/'state/primed_confirmation_progress_v32.json',dict(results=results))
                if not passed:return
            c=json.loads((ROOT/'configs/primed_graph_serving_qwen8_L8000_O16_N64_s3_c28_v31.json').read_text())
            c.update(name=f'primed_graph_serving_qwen8_L8000_O16_N64_s3_c{cut}_v32',layer_cut=cut)
            name=save(c)
            if not execute(name,restore=False):return
            row=dict(name=name,metrics=metrics(name),output_gate=compare(native[0]['name'],name))
            results.append(row)
            write(ROOT/'state/primed_confirmation_progress_v32.json',dict(results=results))
            print('PRIMED_CUT',json.dumps(row),flush=True)
            if not row['output_gate']['passed']:return
        candidates=[r for r in prior+results if 'metrics' in r and re.search(r'_c\d+_',r['name']) and r['output_gate']['passed']]
        safe=[r for r in candidates if r['metrics']['Decode_absolute_SLO_pass'] and all(
            r['metrics'][key]<=1.05*n['metrics'][key] for n in native for key in ('ITL_p99_ms','TPOT_p95_ms'))]
        best=min(safe or candidates,key=lambda r:r['metrics']['TTFT_p95_ms'])
        coloc=next(r for r in prior if 'metrics' in r and '_colocated_' in r['name'])
        selection=dict(selected=best['name'],pilot_noninferiority_against_both_Native_controls=bool(safe),
            Native_A_A_ratios={k:native[1]['metrics'][k]/native[0]['metrics'][k] for k in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms')},
            scope='If no cut passes both Native controls, confirmation remains diagnostic. No decode protection or GO is inferred from absolute SLO alone.')
        write(ROOT/'analysis/primed_confirmation_selection.json',selection)
        templates={k:json.loads((ROOT/'configs'/(r['name']+'.json')).read_text()) for k,r in [('native',native[0]),('layer',best),('colocated',coloc)]}
        assert templates['layer']['layer_mode']=='MIGRATE' and templates['layer']['arm']=='PD_G1'
        assert templates['colocated']['arm']=='DP2'
        groups=[]
        for index,seed in enumerate((2026111231,2026111232,2026111233)):
            rng=random.Random(seed);offsets=[0.]
            for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(native[0]['metrics']['offered_QPS']))
            order=[('native','layer','colocated'),('layer','colocated','native'),('colocated','native','layer')][index]
            group=dict(seed=seed,arms={},order=list(order))
            for label in order:
                c=dict(templates[label],name=f'primed_confirm_qwen8_L8000_O16_N128_s{seed%100}_{label}_v32',
                    count=128,request_caps=[16]*128,arrival_offsets=offsets,arrival_seed=seed,timeout=360,
                    hidden_source_backpressure=label=='layer',phase='EXACT_TEXT_ENABLED_HELDOUT_ARRIVAL_CONFIRMATION',
                    scope='Three balanced paired independent Starts. Same 128 distinct prefixes. Sixty-four prompts appeared in selection. Fixed SLO and text-enabled output. No sustainable-capacity claim.')
                group['arms'][label]=save(c)
            groups.append(group)
        write(ROOT/'state/primed_confirmation_plan_v32.json',dict(groups=groups,selection=selection))
        for group in groups:
            for label in group['order']:
                if time.time()>LONG_START_CUTOFF:return
                name=group['arms'][label]
                if not execute(name,restore=False):return
                row=dict(name=name,metrics=metrics(name));results.append(row)
                write(ROOT/'state/primed_confirmation_progress_v32.json',dict(results=results))
                print('PRIMED_CONFIRM',json.dumps(row),flush=True)
            checks=[compare(group['arms']['native'],group['arms'][label]) for label in ('layer','colocated')]
            record('primed_N128_outputs',dict(seed=group['seed'],checks=checks,passed=all(r['passed'] for r in checks)))
            if not all(r['passed'] for r in checks):return
    finally:restore()


if __name__=='__main__':main()
