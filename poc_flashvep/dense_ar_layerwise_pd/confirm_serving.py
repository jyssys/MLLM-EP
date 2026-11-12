"""Select on N64, then use new paired arrival seeds for N128."""
import fcntl,json,random
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics,compare


def prepare():
    progress=json.loads((ROOT/'state/serving_pilot_progress.json').read_text())['results']
    assert len(progress)==7 and all(x['complete'] for x in progress)
    base_metric=progress[0]['metrics'];eligible=[]
    for row in progress[3:]:
        m=row['metrics'];ok=row['output_gate']['passed'] and m['Decode_absolute_SLO_pass']
        ok=ok and m['ITL_p99_ms']<=1.05*base_metric['ITL_p99_ms'] and m['TPOT_p95_ms']<=1.05*base_metric['TPOT_p95_ms']
        if ok:eligible.append(row)
    if not eligible:
        write(ROOT/'analysis/N128_selection_gate.json',dict(passed=False,reason='NO_DECODE_SAFE_PILOT_CANDIDATE'))
        return []
    best=min(eligible,key=lambda x:x['metrics']['TTFT_p95_ms'])
    write(ROOT/'analysis/N128_selection_gate.json',dict(passed=True,selected=best['name'],
        selection_seed=2026111203,eligible=[x['name'] for x in eligible],
        scope='Point-estimate pilot selection only. Confirmation ratios and uncertainty decide performance.'))
    templates={label:json.loads((ROOT/'configs'/(name+'.json')).read_text()) for label,name in [
        ('native',progress[0]['name']),('span',progress[2]['name']),('layer',best['name'])]}
    jobs=[];groups=[]
    for index,seed in enumerate((2026111211,2026111212,2026111213)):
        rng=random.Random(seed);offsets=[0.]
        for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(templates['native']['arrival_rate']))
        order=[('native','span','layer'),('layer','native','span'),('span','layer','native')][index];group={}
        for label in order:
            c=dict(templates[label],name=f'confirm_qwen8_L8000_O16_N128_s{seed%100}_{label}_v1',
                count=128,request_caps=[16]*128,arrival_seed=seed,arrival_offsets=offsets,
                timeout=360,phase='HELDOUT_ARRIVAL_PAIRED_CONFIRMATION',
                scope='Same 128 distinct prompt IDs. Arrival seed is held out from N64 selection. Only 64 prompts were unseen in selection. Balanced method order. No CUDA markers or state hashing.')
            name=save(c);group[label]=name;jobs.append(name)
        groups.append(dict(seed=seed,arms=group,order=list(order)))
    write(ROOT/'state/serving_confirmation_plan.json',dict(jobs=jobs,groups=groups,selection=best['name']))
    return jobs


def main():
    results=[]
    try:
        for n in prepare():
            ok=execute(n,restore=False);row=dict(name=n,complete=ok)
            if ok:row['metrics']=metrics(n)
            results.append(row);write(ROOT/'state/serving_confirmation_progress.json',dict(results=results))
            print('CONFIRM',n,ok,flush=True)
            if not ok:break
        plan=json.loads((ROOT/'state/serving_confirmation_plan.json').read_text());checks=[]
        for group in plan['groups']:
            a=group['arms'];checks.extend(compare(a['native'],a[label]) for label in ('span','layer'))
        write(ROOT/'analysis/serving_confirmation_output_gates.json',dict(checks=checks,passed=all(x['passed'] for x in checks)))
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
