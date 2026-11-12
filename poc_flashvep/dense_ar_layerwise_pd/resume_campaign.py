"""Resume only qualified stages and retain failed performance gates."""
import fcntl,json,random,time
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics,compare
from .quiet_analysis import main as quiet_analysis


def config(name):return json.loads((ROOT/'configs'/(name+'.json')).read_text())


def restore():
    with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not resources().processes():resources().restore(list(range(8)))


def record(stage,row):
    path=ROOT/'state/resumed_campaign_progress_v23.json'
    progress=json.loads(path.read_text()) if path.exists() else dict(stages={})
    progress['stages'].setdefault(stage,[]).append(row)
    progress.update(updated_epoch=time.time(),GPU_deadline_epoch=GPU_DEADLINE)
    write(path,progress);print(stage,json.dumps(row),flush=True)


def run(name,stage,reference=None,performance=False):
    ok=execute(name,restore=False);row=dict(name=name,complete=ok)
    if ok and reference:row['output_gate']=compare(reference,name)
    if ok and performance:row['metrics']=metrics(name)
    record(stage,row)
    return ok and (not reference or row['output_gate']['passed'])


def calibrate():
    c=config('capacity_qwen8_L8000_O16_p_only_fixed42GiB_B124_s0')
    c.update(name='eager_capacity_qwen8_L8000_O16_fixed42GiB_B124_v23',
        ar_enforce_eager=True,ar_execution_family='BI_EAGER_NO_GRAPH',
        phase='SAME_FAMILY_LOCAL_CAPACITY',jobs=[dict(batch=b,cap=16,repeats=5) for b in (1,2,4)])
    name=save(c)
    return name if run(name,'capacity') else None


def physical(capacity):
    template=config('eager_serving_qwen8_L8000_O16_N64_s3_guard_v22')
    for seed in (2026111224,2026111225,2026111226):
        rng=random.Random(seed);offsets=[0.]
        for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(template['arrival_rate']))
        names=[]
        for label,observed in [('cleanA',False),('quiet',True),('cleanB',False)]:
            c=dict(template,name=f'eager_physical_qwen8_L8000_O16_N128_s{seed%100}_{label}_v23',
                count=128,request_caps=[16]*128,arrival_offsets=offsets,arrival_seed=seed,
                runtime_observation=observed,runtime_observation_roles=['prefill'],
                physical_activity=observed,activity_roles=['prefill','decode'],
                activity_quiet_destination=True,activity_host_pool_bytes=1024**3,
                activity_memcpy=False,clean_timing=not observed,timeout=360,
                phase='EAGER_EXACT_NATIVE_MATCHED_CENSORED_PHYSICAL_REPLICATION',
                scope='Same exact-BF16 family as actual Layer continuation. Quiet-D capture observes only empty-engine epochs. Uncaptured D time remains unknown. Five-metric 5% interference screen is mandatory.')
            names.append(save(c))
        plan=f'eager_physical_s{seed%100}_plan_v23.json'
        write(ROOT/'state'/plan,dict(jobs=names,capacity_job=capacity))
        if not all(run(n,'physical',performance=True) for n in names):return False
        output=f'eager_physical_s{seed%100}_screen_v23.json';quiet_analysis(plan,output)
        screen=json.loads((ROOT/'analysis'/output).read_text())
        record('physical_gates',dict(seed=seed,trace_qualified=screen['trace_qualified'],
            P_gate=screen['P_gate'],Decode_absolute_SLO_pass=screen['Decode_absolute_SLO_pass'],analysis=output))
    return True


def small_batch():
    template=config('eager_serving_qwen8_L8000_O16_N64_s3_native_v22')
    full32='eager_qwen8_L8000_O16_N32_independent_full_cohort_reference_v21'
    qual=config(full32)
    for count in (8,32):
        c=dict(qual,name=f'eager_colocated_full_cohort_N{count}_qual_v23',
            count=count,request_caps=[16]*count,arrival_offsets=[0.]*count,
            colocated_cohorts=True,D_admission_batch=4,D_cohort_collect_ms=2,
            phase='EAGER_STRONG_COLOCATED_OUTPUT_QUALIFICATION')
        n=save(c)
        ok=execute(n,restore=False)
        check=compare(full32,n,True) if ok else dict(passed=False)
        record('colocated_qualification',dict(name=n,complete=ok,output_gate=check))
        if not ok or not check['passed']:return False
    arms=[('guard2',None,32,'PD_G1'),('c28_guard2','MIGRATE',28,'PD_G1'),
        ('colocated_cohort2',None,32,'DP2')]
    for label,mode,cut,arm in arms:
        c=dict(template,name=f'eager_serving_qwen8_L8000_O16_N64_s3_{label}_v23',
            layer_mode=mode,layer_cut=cut,arm=arm,serialize_D_imports=arm!='DP2',
            cohort_barrier=arm!='DP2',D_admission_batch=2,colocated_cohorts=arm=='DP2',
            separate_suffix_runner=mode=='MIGRATE',partial_KV_wire=mode=='MIGRATE',
            reuse_prefill_token=arm!='DP2' and mode!='MIGRATE',
            hidden_transport=None if arm=='DP2' else 'NIXL_GPU',
            phase='SMALLER_COHORT_DIAGNOSTIC_SELECTION',
            scope='Common two-request cohort cap for Native, Layer, and full-replica controls. No SLO change. Selection results do not confirm a benefit.')
        if not run(save(c),'small_batch_selection',template['name'],True):return False
    return True


def diagnostic():
    names=[]
    for label in ('guard','c35','c28'):
        c=config(f'eager_serving_qwen8_L8000_O16_N64_s3_{label}_v22')
        c.update(name=f'eager_qwen8_L8000_O16_N32_{label}_event_diagnostic_v23',
            count=32,request_caps=[16]*32,arrival_offsets=[0.]*32,
            runtime_observation=True,runtime_observation_roles=['prefill','decode'],
            execution_events=True,verify_layer_execution=True,state_capture=False,
            clean_timing=False,phase='DECODE_REGRESSION_EXECUTION_ENCLOSURE_DIAGNOSTIC',
            scope='CUDA-event enclosures and Host scheduling observations. Not kernel union times. Extra observation may change latency and never supplies clean performance.')
        names.append(save(c))
    for n in names:
        if not run(n,'execution_diagnostic',performance=True):return False
    return True


def confirmation():
    native='eager_serving_qwen8_L8000_O16_N64_s3_native_v22'
    pilot=json.loads((ROOT/'state/eager_serving_pilot_progress.json').read_text())['results']
    candidates=[r for r in pilot if '_c28_' in r['name'] or '_c32_' in r['name']]
    for label in ('c28_guard2',):
        n=f'eager_serving_qwen8_L8000_O16_N64_s3_{label}_v23'
        if (ROOT/'raw'/n/'dispatcher_result.json').exists():
            candidates.append(dict(name=n,metrics=metrics(n),output_gate=compare(native,n)))
    baseline=metrics(native)
    eligible=[r for r in candidates if r['output_gate']['passed'] and r['metrics']['Decode_absolute_SLO_pass']
        and r['metrics']['ITL_p99_ms']<=1.05*baseline['ITL_p99_ms']
        and r['metrics']['TPOT_p95_ms']<=1.05*baseline['TPOT_p95_ms']]
    pool=eligible or [r for r in candidates if r['output_gate']['passed'] and r['metrics']['Decode_absolute_SLO_pass']]
    if not pool:return False
    selected=min(pool,key=lambda r:r['metrics']['TTFT_p95_ms'])['name']
    colocated=['eager_serving_qwen8_L8000_O16_N64_s3_colocated_cohort_v22',
        'eager_serving_qwen8_L8000_O16_N64_s3_colocated_cohort2_v23']
    colocated=[n for n in colocated if metrics(n).get('complete')]
    colocated=min(colocated,key=lambda n:metrics(n)['TTFT_p95_ms'])
    write(ROOT/'analysis/eager_confirmation_selection_v23.json',dict(selected=selected,
        colocated=colocated,pilot_decode_noninferiority_pass=bool(eligible),
        scope='If the pilot protection gate fails, N128 runs are bounded diagnostic confirmation. They cannot establish a qualified Phase D success without the original exactness and Decode gates.'))
    templates=dict(native=config(native),layer=config(selected),colocated=config(colocated))
    groups=[]
    for index,seed in enumerate((2026111211,2026111212,2026111213)):
        rng=random.Random(seed);offsets=[0.]
        for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(templates['native']['arrival_rate']))
        order=[('native','layer','colocated'),('layer','colocated','native'),('colocated','native','layer')][index]
        group=dict(seed=seed,arms={},order=list(order))
        for label in order:
            c=dict(templates[label],name=f'eager_confirm_qwen8_L8000_O16_N128_s{seed%100}_{label}_v25',
                count=128,request_caps=[16]*128,arrival_seed=seed,arrival_offsets=offsets,
                timeout=360,hidden_source_backpressure=label=='layer',
                phase='PAIRED_HELDOUT_ARRIVAL_CONFIRMATION' if eligible else 'BOUNDED_STATIC_DIAGNOSTIC_CONFIRMATION',
                scope='Three balanced independent Server starts per arm. Same 128 distinct prompts and held-out arrival seeds. Sixty-four prompts occurred in selection. Marker-free. Failed pilot protection remains a recorded failure.')
            n=save(c);group['arms'][label]=n
        groups.append(group)
    write(ROOT/'state/eager_confirmation_plan_v23.json',dict(groups=groups,selection=selected))
    for g in groups:
        for label in g['order']:
            if not run(g['arms'][label],'N128_confirmation',performance=True):return False
        checks=[compare(g['arms']['native'],g['arms'][label]) for label in ('layer','colocated')]
        record('N128_outputs',dict(seed=g['seed'],checks=checks,passed=all(x['passed'] for x in checks)))
        if not all(x['passed'] for x in checks):return False
    return True


def main():
    preserve()
    pilot=ROOT/'state/eager_serving_pilot_progress.json'
    assert len(json.loads(pilot.read_text())['results'])==7
    try:
        capacity=calibrate()
        if not capacity:return
        physical(capacity)
        if not small_batch():return
        diagnostic()
        confirmation()
    finally:restore()


if __name__=='__main__':main()
