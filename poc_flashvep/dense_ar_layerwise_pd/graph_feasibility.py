"""Test exact opaque operators with Native Decode graphs, not compiler fusion."""
import fcntl,json
from .common import *
from .prepare import save
from .run import execute
from .analyze import compare,metrics
from .state_qualification import state_compare,events


def graph_config(c,name):
    return dict(c,name=name,ar_enforce_eager=False,ar_compile_mode=0,
        ar_custom_ops=['all'],ar_cudagraph_mode='FULL_DECODE_ONLY',
        decode_only_graph_roles=['decode','replica0','replica1'],
        ar_execution_family='BI_OPAQUE_NO_COMPILE_DECODE_GRAPH',graph_replay_audit=True,
        hidden_source_backpressure=c.get('layer_mode')=='MIGRATE',
        minimal_geometry_counter=True,
        source_token_release_policy='AFTER_IMPORT_SUBMIT',
        phase='EXACT_OPAQUE_OPERATOR_NATIVE_GRAPH_FEASIBILITY',
        scope='Compiler mode zero preserves explicit BF16 operator boundaries. Native Decode graphs are a generic runtime optimization offered to all arms. The actual first P token stays held until valid import and next full Decode model or Graph submission. No SLO changes.')


def gate(name,count,migration=False):
    cut=json.loads((ROOT/'configs'/(name+'.json')).read_text())['layer_cut']
    full32='eager_qwen8_L8000_O16_N32_independent_full_cohort_reference_v21'
    checks=[compare(full32,name,True)]
    dest=events(name,'decode')
    captures=[r for r in dest if r['event']=='CAPTURED_FULL_GRAPH_LAYERS']
    checks.append(dict(passed=bool(captures) and all(r['passed'] and r['actual']==list(range(36)) for r in captures),
        graph_capture_definitions=len(captures),scope='Observed original full-36-layer Decode graph definitions.'))
    if migration:
        reference=f'eager_qwen8_L8000_O16_N{count}_reference_qual_v16'
        checks.extend([state_compare(reference,name,'prefill',cut),state_compare(reference,name,'decode')])
        source=events(name,'prefill');dest=events(name,'decode')
        prompt_counts={f'q{i}':0 for i in range(count)};decode_counts=dict(prompt_counts)
        for r in dest:
            if r['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and r.get('measured'):
                expected=list(range(cut,36)) if r['prefilling'] else list(range(36))
                assert r['actual']==expected and r['passed']
                for rid in r['requests']:(prompt_counts if r['prefilling'] else decode_counts)[rid]+=1
            elif r['event']=='FULL_GRAPH_REPLAY':
                for rid in r['requests']:
                    if rid in decode_counts:
                        assert not r['prefilling'];decode_counts[rid]+=1
        source_counts={f'q{i}':0 for i in range(count)}
        for r in source:
            if r['event']=='ACTUAL_ATTENTION_LAYER_EXECUTION' and r.get('measured'):
                assert r['prefilling'] and r['actual']==list(range(cut)) and r['passed']
                for rid in r['requests']:source_counts[rid]+=1
        checks.append(dict(passed=all(x==1 for x in prompt_counts.values()) and all(x==1 for x in source_counts.values())
            and all(x==15 for x in decode_counts.values()),source_prompt_counts=source_counts,
            destination_suffix_counts=prompt_counts,destination_full_Decode_counts=decode_counts,
            scope='Measured Prefix/Suffix backend calls and actual full-model graph replay calls. No duplicate or missing Layer work.'))
    for role in ('prefill','decode'):
        life=json.loads((ROOT/'raw'/name/f'KV_lifetime_{role}.json').read_text())
        checks.append(life)
    result=dict(name=name,checks=checks,passed=all(x['passed'] for x in checks),evidence='CORRECTNESS_ONLY')
    write(ROOT/'analysis'/(name+'_graph_gate.json'),result)
    print('GRAPH_GATE',name,result['passed'],flush=True);return result['passed']


def main(version='v28'):
    # Start this module only after the active Resume campaign finishes.
    jobs=[]
    for count in (8,32):
        c=json.loads((ROOT/'configs'/f'eager_qwen8_L8000_O16_N{count}_native_token_reuse_prefix_qual_v20.json').read_text())
        c=graph_config(c,f'graph_qwen8_L8000_O16_N{count}_native_qual_{version}')
        c.update(execution_events=False,state_capture=False,verify_layer_execution=True)
        jobs.append((save(c),count,False))
        for cut in (35,28):
            c=json.loads((ROOT/'configs'/f'eager_qwen8_L8000_O16_N{count}_c{cut}_qual_v16.json').read_text())
            c=graph_config(c,f'graph_qwen8_L8000_O16_N{count}_c{cut}_qual_{version}')
            c.update(execution_events=False,verify_layer_execution=True)
            jobs.append((save(c),count,True))
    write(ROOT/'state'/f'graph_feasibility_plan_{version}.json',dict(jobs=[r[0] for r in jobs]))
    results=[]
    try:
        for name,count,migration in jobs:
            complete=execute(name,restore=False)
            passed=gate(name,count,migration) if complete else False
            results.append(dict(name=name,complete=complete,passed=passed))
            write(ROOT/'state'/f'graph_feasibility_progress_{version}.json',dict(results=results))
            if not passed:return
        template=json.loads((ROOT/'configs/eager_serving_qwen8_L8000_O16_N64_s3_native_v22.json').read_text())
        for label,mode,arm in [('native',None,'PD_G1'),('c35','MIGRATE','PD_G1'),('c28','MIGRATE','PD_G1'),('colocated',None,'DP2')]:
            c=graph_config(template,f'graph_serving_qwen8_L8000_O16_N64_s3_{label}_{version}')
            c.update(layer_mode=mode,layer_cut=35 if label=='c35' else 28,arm=arm,
                serialize_D_imports=arm!='DP2',cohort_barrier=arm!='DP2',
                partial_KV_wire=mode=='MIGRATE',separate_suffix_runner=mode=='MIGRATE',
                reuse_prefill_token=arm!='DP2' and mode!='MIGRATE',
                hidden_transport=None if arm=='DP2' else 'NIXL_GPU',colocated_cohorts=arm=='DP2',
                execution_events=False,state_capture=False,verify_layer_execution=False,clean_timing=True)
            name=save(c)
            if not execute(name,restore=False):return
            check=compare('eager_serving_qwen8_L8000_O16_N64_s3_native_v22',name)
            row=dict(name=name,metrics=metrics(name),output_gate=check)
            results.append(row);write(ROOT/'state'/f'graph_feasibility_progress_{version}.json',dict(results=results))
            print('GRAPH_PILOT',json.dumps(row),flush=True)
            if not check['passed']:return
    finally:
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
