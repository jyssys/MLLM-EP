"""Keep final composition correctness separate from paired performance."""
import argparse
import json
import time
from .common import ROOT,write,dispatch_allowed
from .run import execute
from .qualification import audit
from .decode_cap_discovery import full_native_state
from .compute_throttle_qualification import no_mixed
from .quality_gate import compare as quality
from .metrics import summarize
from .final_confirmation_jobs import signature


def qualify(lane,plan):
    results=[]
    for key in plan['qualification_lanes'][lane]:
        item=plan['qualification'][key]
        name=item['job']
        try:
            assert execute(name),name
            cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
            assert signature(cfg)==item['signature']
            if cfg.get('axis_kind'):
                state=audit(name)
                assert state['passed'] and state['decode_full36_observation_complete']
                assert all(v['status']=='PASS' for v in state['imported_prefix_KV'])
            else:
                state=full_native_state(name,cfg.get('whole_prefill_deflection'))
            score=quality(name)
            assert score['passed'],score
            proof=dict(passed=True,full400_passed=True,name=name,signature=item['signature'],
                source_capsule_SHA256=plan['source_capsule_SHA256'],state=state,quality=score,
                compute_rule=no_mixed(name) if cfg.get('d_compute_throttle') else None,
                evidence='CORRECTNESS_ONLY',scope='Fresh final-source full400 composition stress. '
                    'Independent strict N1/N8 teacher proofs remain inherited and separately scoped. '
                    'This is not a performance or decode-tail confidence result.')
        except Exception as error:
            proof=dict(passed=False,name=name,error=repr(error),evidence='FAILED',
                scope='Preserve the failure. Do not dispatch this composition as qualified performance.')
        write(ROOT/'state'/item['gate'],proof)
        results.append(proof)
        print('FINAL_COMPOSITION',name,proof['passed'],flush=True)
    write(ROOT/'state'/f'final_qualification_lane{lane}_complete_v1.json',dict(complete=True,results=results))


def performance(lane,plan):
    results=[]
    for name in plan['performance_lanes'][lane]:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        gate=cfg.get('final_composition_gate')
        if gate:
            path=ROOT/'state'/gate
            while not path.exists():
                assert dispatch_allowed()
                time.sleep(5)
            proof=json.loads(path.read_text())
            if not proof.get('passed'):
                results.append(dict(name=name,status='NOT_RUN',reason='BLOCKED_CORRECTNESS',gate=gate))
                write(ROOT/'state'/f'final_performance_lane{lane}_progress_v1.json',dict(results=results))
                continue
        success=execute(name)
        if success:
            m=summarize(name)
            score=quality(name)
            if cfg.get('axis_kind'):
                coverage=audit(name)
                assert coverage['passed'] and coverage['decode_full36_observation_complete']
            result=dict(name=name,status='MEASURED_ONLINE',quality=score,
                TTFT_p95_ms=m['TTFT_p95_ms'],ITL_p99_ms=m['ITL_p99_ms'],TPOT_p95_ms=m['TPOT_p95_ms'],
                SLO_goodput=m['SLO_qualified_output_tokens_per_second'])
        else:
            result=dict(name=name,status='FAILED')
        results.append(result)
        write(ROOT/'state'/f'final_performance_lane{lane}_progress_v1.json',dict(results=results))
        print('FINAL_SERVING',result,flush=True)
    write(ROOT/'state'/f'final_performance_lane{lane}_complete_v1.json',dict(complete=True,results=results))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('qualify','performance'))
    parser.add_argument('lane',type=int,choices=range(4))
    args=parser.parse_args()
    plan=json.loads((ROOT/'configs/final_continuous_execution_plan_v1.json').read_text())
    (qualify if args.phase=='qualify' else performance)(args.lane,plan)


if __name__=='__main__':
    main()
