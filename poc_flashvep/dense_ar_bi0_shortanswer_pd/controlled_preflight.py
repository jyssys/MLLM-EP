"""Exercise conditional Native fallback and full transport ownership first."""
import json
import time
from .common import ROOT,write,save
from .qualification import configuration
from .run import execute
from .quality import quality_gate
from .serving_analysis import analyze


def main(tag='v1',block=16,restore=False,asynchronous=False,qualification_tag=None):
    axis_flag=(f'axis_qualification_{qualification_tag}_complete.json' if qualification_tag else
        'axis_qualification_complete.json' if tag=='v1' else 'axis_qualification_shared128_v2_complete.json')
    while not (ROOT/'state'/axis_flag).exists():
        for marker in ('axis_qualification_failed.json','axis_state_mismatch.json','axis_qualification_hold.json','STOP_REQUESTED.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(3)
    declaration=ROOT/'configs'/f'frozen_controlled_preflight_{tag}.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),N=8,arrival_offsets=[0.]*8,source_credit=4,actual_batches=[1,1,1],cut=4,
            arms=['Native','StaticLayer','StaticToken','WindowLayer','WindowToken'],
            KV_block_tokens=block,restore_official_UCX_init=restore,async_scheduling=asynchronous,
            scope='Burst correctness control only. Every D admission needs a device-wide idle certificate. '
            'Source never waits for a D reservation ACK before normal Native computation. No active-decode gap insertion.'))
    results=[]
    for label,axis,mode in (('Native',None,'STATIC'),('StaticLayer','Layer','STATIC'),('StaticToken','Token','STATIC'),
                           ('WindowLayer','Layer','WINDOW'),('WindowToken','Token','WINDOW')):
        name=f'qual_Controlled_{label}_N8_BI0_{tag}'
        extra=dict(axis_kind=axis,axis_cut=4,axis_state_capture=True,hidden_source_slots=4) if axis else {}
        cfg=configuration(name,axis or 'Native','qualification8.json',serial=False,arrival_offsets=[0.]*8,
            state_capture=not bool(axis),numerical_probe=True,controlled_pipeline=True,axis_mode=mode,
            source_credit_limit=4,execution_family='MATCHED_NO_COMPILE',async_scheduling=asynchronous,
            numerical_probe_sync=asynchronous,async_qualification_only=asynchronous,
            KV_block_tokens=block,restore_official_UCX_init=restore,phase='CONTROLLED_BURST_CORRECTNESS',**extra)
        assert execute(save(cfg)),name
        quality=quality_gate(name)
        certs=[json.loads(line) for line in (ROOT/'raw'/name/'idle_certificates.jsonl').read_text().splitlines()]
        queries={}
        for role in ('prefill','decode'):
            queries[role]=json.loads((ROOT/'raw'/name/f'actual_queries_{role}.json').read_text())['rows']
        batch_pass=all(r['requests']==1 for rows in queries.values() for r in rows)
        imported=True
        if axis:
            states={role:json.loads((ROOT/'raw'/name/f'axis_events_{role}.json').read_text())['states'] for role in ('prefill','decode')}
            imported=all(states['prefill'][f'q{i}']['KV']==states['decode'][f'import_q{i}']['KV'] for i in range(8))
        lifetime={p.name:json.loads(p.read_text())['passed'] for p in (ROOT/'raw'/name).glob('*lifetime*.json')}
        decisions=[json.loads(line) for line in (ROOT/'raw'/name/'axis_decisions.jsonl').read_text().splitlines()]
        choices={r['action']['kind'] for r in decisions}
        branch_pass=axis is None or mode=='STATIC' or choices=={'Native',axis}
        passed=batch_pass and imported and all(lifetime.values()) and bool(certs) and all(r['passed'] for r in certs) and branch_pass
        results.append(dict(name=name,passed=passed,actual_batches_matched=batch_pass,imported_full_valid_KV_exact=imported,
            certificates=len(certs),choices=sorted(choices),lifetime=lifetime,quality=quality,
            evidence='CORRECTNESS_ONLY',scope='CPU state copies invalidate timings. Full200 task quality remains unqualified.'))
        write(ROOT/'analysis'/f'controlled_preflight_{tag}.json',dict(results=results))
        assert passed,name
    gate='controlled_axis_structural_gate.json' if tag=='v1' else f'controlled_axis_structural_gate_{tag}.json'
    write(ROOT/'state'/gate,dict(time=time.time(),passed=True,qualified_cuts=[4],
        source_credit=4,actual_batches=[1,1,1],scope='Quiescent-only controlled continuation. Not active-decode window scheduling. Full200 performance/quality not yet qualified.'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/controlled_preflight_hold.json',dict(time=time.time(),error=repr(error)));raise
