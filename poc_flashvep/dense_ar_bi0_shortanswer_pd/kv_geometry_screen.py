"""Change KV block geometry only after full-state transport qualification."""
import json
import time
from .common import ROOT,write,save
from .qualification import configuration
from .run import execute
from .campaign import serving
from .metrics import summarize
from .quality import quality_gate
from .serving_analysis import analyze


def state_gate(block):
    names={label:f'qual_{label}_KV{block}_ShortAnswer_N8_BI0_v1' for label in ('Standard','Native')}
    states={}
    for label,name in names.items():
        folder=ROOT/'raw'/name
        states[label]={p.stem.removeprefix('prompt_state_'):json.loads(p.read_text())['requests']
                       for p in folder.glob('prompt_state_*.json')}
    rows=[]
    for rid,teacher in states['Standard']['single'].items():
        if not rid.startswith('q'):continue
        source=states['Native']['prefill'][rid];dest=states['Native']['decode'][rid]
        rows.append(dict(request_id=rid,full_KV_exact=teacher['KV']==source['KV']==dest['KV'],
            P_hidden_exact=teacher['final_hidden_SHA256']==source['final_hidden_SHA256'],
            D_query_correct=dest['first_observed_query']==source['original_input_tokens']-1))
    quality={label:quality_gate(name) for label,name in names.items()}
    passed=len(rows)==8 and all(all(r[k] for k in ('full_KV_exact','P_hidden_exact','D_query_correct')) for r in rows)
    passed=passed and all(q['point_passed'] for q in quality.values())
    write(ROOT/'analysis'/f'KV{block}_N8_state_gate.json',dict(passed=passed,rows=rows,quality=quality,
        evidence='CORRECTNESS_ONLY',scope='Same block geometry reference and exact Native transfer. N8 does not certify full200 task quality.'))
    return passed


def main():
    design=ROOT/'configs/frozen_BI0_KV_geometry_screen_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),block_tokens=[128,256],original_block_tokens=16,
            KV_budget_GiB=42,B=8192,S=1024,NIXL_threads=4,arrival_seed=2026112033,
            hypothesis='Larger valid KV blocks reduce native descriptor count and host posting work.',
            competing_hypothesis='Bytes, device dependencies or scheduler effects dominate despite fewer descriptors.',
            gates='N8 fullKV/hidden/reference, N32 quality, then full200 diagnostic. Do not replace default Main results.',
            scope='Stock supported FlashAttention block geometries. Report memory rounding and actual batch changes.'))
    results=[]
    for block in (128,256):
        for label,extra in (('Standard',dict(reference_single=True,gpus=[1])),('Native',{})):
            cfg=configuration(f'qual_{label}_KV{block}_ShortAnswer_N8_BI0_v1','Native','qualification8.json',
                KV_block_tokens=block,**extra)
            if not execute(save(cfg)):
                write(ROOT/'state'/f'KV{block}_unsupported.json',dict(time=time.time(),status='FAILED',
                    scope='Preserve failed geometry. Do not use it as valid performance.'))
                break
        else:
            if not state_gate(block):continue
            # This seed and order match the original N32 four-thread screen.
            name=f'diag_Native_KV{block}_ShortAnswer_N32_q2.35_BI0_v1'
            result=serving(name,'Native','qualification32.json',32,2.35,2026112033,
                KV_block_tokens=block,phase='KV_GEOMETRY_DIAGNOSTIC')
            analyze(name);results.append(result)
            write(ROOT/'analysis/BI0_KV_geometry_screen_v1.json',dict(results=results))
            if not result['quality_gate']['point_passed']:continue
            name=f'diag_Native_KV{block}_ShortAnswer200_q2.35_s2026112051_BI0_v1'
            result=serving(name,'Native','longbench_shortanswer200_v2.json',200,2.35,2026112051,
                KV_block_tokens=block,phase='KV_GEOMETRY_FULL200_DIAGNOSTIC')
            analyze(name);results.append(result)
            write(ROOT/'analysis/BI0_KV_geometry_screen_v1.json',dict(results=results))
    write(ROOT/'state/KV_geometry_screen_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/KV_geometry_screen_hold.json',dict(time=time.time(),error=repr(error)));raise
