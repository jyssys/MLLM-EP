"""Qualify a fast candidate only after the one-factor native screen."""
import argparse
import json
import time
from .common import ROOT,write


def main(teacher=False):
    while not (ROOT/'state/async_screen_complete.json').exists():
        for marker in ('STOP_REQUESTED.json','async_screen_hold.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(3)
    candidate=json.loads((ROOT/'state/async_screen_complete.json').read_text())
    if not candidate['faster_candidate']:
        write(ROOT/'state/async_candidate_not_adopted.json',dict(time=time.time(),screen=candidate,
            scope='No automatic Layer/Token performance dispatch after a failed speed screen.'));return
    if teacher:
        from .token_geometry_campaign import main as qualify
        qualify(tag='fast128_v3',qualification_tag='fast128_v3',asynchronous=True)
    else:
        declaration=ROOT/'configs/frozen_fast_shared_runtime_v3.json'
        if not declaration.exists():
            write(declaration,dict(time=time.time(),B=8192,S=1024,KV_GiB=42,KV_block_tokens=128,
                restore_official_UCX_init=True,NIXL_threads=4,async_scheduling=True,
                execution='Compile0, full decode graph',actual_batches=[1,1,1],source_credit=4,
                scope='Conditional candidate after a Native-only one-factor speed screen. '
                'All output/state/lifetime and same-geometry Token gates must pass before performance. '
                'The slower async-off family remains separate evidence.'))
        from .axis_qualification import main as qualify
        qualify(tag='fast128_v3',cuts=(4,8,12,1,2,6),block=128,restore=True,asynchronous=True)
        assert (ROOT/'state/axis_qualification_fast128_v3_complete.json').exists()
        from .qualification import configuration
        from .run import execute
        from .common import save
        name='qual_FastControlledNative_N8_BI0_v3'
        cfg=configuration(name,'Native','qualification8.json',serial=False,controlled_pipeline=True,
            source_credit_limit=4,execution_family='MATCHED_NO_COMPILE',async_scheduling=True,
            KV_block_tokens=128,restore_official_UCX_init=True,numerical_probe=True,numerical_probe_sync=True)
        assert execute(save(cfg)),name
        teacher_name='qual_Standard_NoCompile_ShortAnswer_N8_BI0_fast128_v3'
        teacher=json.loads((ROOT/'raw'/teacher_name/'prompt_state_single.json').read_text())['requests']
        p=json.loads((ROOT/'raw'/name/'prompt_state_prefill.json').read_text())['requests']
        d=json.loads((ROOT/'raw'/name/'prompt_state_decode.json').read_text())['requests']
        rows=[dict(request_id=f'q{i}',full_KV_exact=teacher[f'q{i}']['KV']==p[f'q{i}']['KV']==d[f'q{i}']['KV'],
            P_hidden_exact=teacher[f'q{i}']['final_hidden_SHA256']==p[f'q{i}']['final_hidden_SHA256']) for i in range(8)]
        certs=[json.loads(line) for line in (ROOT/'raw'/name/'idle_certificates.jsonl').read_text().splitlines()]
        lifetime={p.name:json.loads(p.read_text())['passed'] for p in (ROOT/'raw'/name).glob('*lifetime*.json')}
        passed=all(r['full_KV_exact'] and r['P_hidden_exact'] for r in rows) and len(certs)==8 and all(c['passed'] for c in certs) and bool(lifetime) and all(lifetime.values())
        write(ROOT/'analysis/fast_native_control_state_v3.json',dict(passed=passed,rows=rows,lifetime=lifetime,evidence='CORRECTNESS_ONLY'))
        assert passed
        write(ROOT/'state/fast_native_control_state_v3_complete.json',dict(time=time.time(),passed=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--teacher',action='store_true');args=parser.parse_args()
    try:main(args.teacher)
    except BaseException as error:
        write(ROOT/'state'/f'async_qualification_{"teacher" if args.teacher else "state"}_hold.json',dict(time=time.time(),error=repr(error)));raise
