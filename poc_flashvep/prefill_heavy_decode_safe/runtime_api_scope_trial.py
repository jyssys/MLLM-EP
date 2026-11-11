"""Check Runtime API correlation before declaring graph timing unavailable."""
import json,time
from .common import *
from .continue_measurement_alternatives import sequence
from .envelope_analysis import analyze
from .host_screen import screen


def main():
    initialize();old=json.loads((ROOT/'state/graph_causal_scope_18k_plan.json').read_text());jobs=[]
    for parent in old['all_jobs']:
        cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text());name=parent.replace('graph_causal_scope_','graph_runtime_scope_')
        cfg.update(name=name,phase='RUNTIME_AND_DRIVER_GRAPH_CORRELATION_SCOPE_CONTROL')
        if parent==old['state8']:cfg['required_qualification']=old['state8']
        elif cfg.get('discovery_activity'):cfg['required_qualification']=old['state8'].replace('graph_causal_scope_','graph_runtime_scope_')
        save(cfg);jobs.append(name)
    path=ROOT/'state/graph_runtime_scope_18k_plan.json'
    write(path,dict(state8=jobs[0],all_jobs=jobs,clean_reference=jobs[1],jobs=jobs[2:],
        scope='The preceding trace contained no matching Driver graph-launch records. This trial also retains Runtime API correlation. No unresolved graph time is discarded.'))
    entry=dict(run=False)
    if sequence(jobs,'graph_runtime_scope_18k_controller'):
        value=analyze(path);t=value['trace']['prefill'];host=screen(jobs[2],window=(t['left'],t['right']))
        entry.update(observer_qualified=value['qualified'],host=host,
            run=value['qualified'] and t['occupied_union_percent']>=80 and host['queue']['half_growth']>0
                and host['queue']['maximum']>=2 and host['P_service_fraction']>=.85 and host['metrics']['Decode_SLO_screen'])
    write(ROOT/'state/graph_runtime_scope_18k_entry.json',entry)
    write(ROOT/'state/graph_runtime_scope_18k_complete.json',dict(complete=True,time=time.time()))


if __name__=='__main__':main()
