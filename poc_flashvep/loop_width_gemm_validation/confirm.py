"""Stop an unsafe repeated arm. Retain every measured result."""
from .common import *
from .run import execute
from .quantum_contract import audit


def main():
    initialize();plan=json.loads((ROOT/'state/confirmation_selection.json').read_text());records=[];blocked={}
    from ..huginn_net_benefit.common import analysis
    from ..looped_pd_atomic_prefill.evidence import full_work
    from ..dynamic_loop_split_serving.lifetime_gate import check
    from .serving_analysis import analyze
    try:
        for name in plan['jobs']:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());key=f'{cfg["model"]}_O{cfg["cap"]}_{cfg["width_arm"]}'
            if key in blocked:
                records.append(dict(name=name,status='NOT_RUN_AFTER_UNSAFE_DECODE',reason=blocked[key]));continue
            print('CONFIRM_START',name,flush=True)
            complete=execute(name);assert complete,('FAILED_CONFIRMATION',name)
            metric=analysis().metrics(name);work=full_work(name);lifetime=check(name)
            assert work['passed'] and lifetime['passed']
            if cfg.get('require_realized_quantum_contract'):assert audit(name)['passed']
            if cfg['width_arm']!='zero' and (not metric['Decode_SLO_screen'] or metric['ITL_limit_misses']):
                blocked[key]=dict(Decode_SLO_screen=metric['Decode_SLO_screen'],ITL_limit_misses=metric['ITL_limit_misses'],name=name)
            records.append(dict(name=name,status='COMPLETE',metrics=metric,work=work,lifetime=lifetime))
            # Balanced order can put the baseline later in the same seed.
            # Compare exact output as soon as that baseline exists.
            for entry in records:
                if entry.get('status')!='COMPLETE' or entry.get('output_checked'):continue
                c=json.loads((ROOT/'configs'/(entry['name']+'.json')).read_text())
                reference=entry['name'].replace('_'+c['width_arm']+'_','_zero_',1)
                if not (ROOT/'raw'/reference/'dispatcher_result.json').exists():continue
                qualified=analyze(entry['name']);assert qualified['exact_output_work_lifetime_passed']
                entry['output_checked']=True
            write(ROOT/'state/confirmation_progress.json',dict(results=records,blocked_arms=blocked,closed=False))
            print('CONFIRM_END',name,metric['TTFT_p95_ms'],metric['ITL_p99_ms'],flush=True)
    finally:
        from ..looped_pd_autonomous_headroom.common import resources
        owner=resources()
        if not owner.processes():owner.restore(list(range(8)))
        write(ROOT/'state/confirmation_progress.json',dict(results=records,blocked_arms=blocked,closed=True,time=time.time()))


if __name__=='__main__':main()
