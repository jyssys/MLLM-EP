"""Resolve an instrumentation mismatch before any transport timing claim."""
import json
import time
from .common import ROOT,save,write
from .qualification import configuration,records
from .run import execute
from .campaign import serving


def main():
    original='qual_Native_ExactCoalesce_KV16_N8_BI0_v1'
    name='qual_Native_NoCoalesce_Synchronized_KV16_N8_BI0_v2'
    design=ROOT/'configs/frozen_coalescing_matched_instrumentation_v2.json'
    if not design.exists():
        write(design,dict(time=time.time(),coalesced=original,control=name,
            reason='Historical control lacked the synchronized numerical probe. New producer hashes differ before transport, while greedy8/8 and current full import8/8 match.',
            intervention='Use the identical synchronized state-capture contract in both Native arms. Only coalescing changes.',
            gate='All full producer KV, hidden, imported KV, greedy outputs and lifetime must match. Otherwise no N32 timing.',
            scope='Keep the original failed strict comparison. Do not attribute a producer-side mismatch to KV transport without this control.'))
    cfg=configuration(name,'Native','qualification8.json',restore_official_UCX_init=True,
        nixl_manual_coalesce=False,numerical_probe=True,numerical_probe_sync=True,
        phase='COALESCING_MATCHED_INSTRUMENTATION_CONTROL')
    assert execute(save(cfg)),name
    folders={True:ROOT/'raw'/original,False:ROOT/'raw'/name}
    states={key:{role:json.loads((folder/f'prompt_state_{role}.json').read_text())['requests'] for role in ('prefill','decode')} for key,folder in folders.items()}
    outputs={key:{r['sample_id']:r['output_ids'] for r in records(folder/'requests.jsonl')} for key,folder in folders.items()}
    rows=[]
    for i in range(8):
        rid=f'q{i}';a=states[True]['prefill'][rid];b=states[False]['prefill'][rid]
        rows.append(dict(request_id=rid,producer_full_KV_exact=a['KV']==b['KV'],
            producer_hidden_exact=a['final_hidden_SHA256']==b['final_hidden_SHA256'],
            both_imports_exact=all(states[k]['prefill'][rid]['KV']==states[k]['decode'][rid]['KV'] for k in (True,False)),
            greedy_exact=outputs[True][i]==outputs[False][i]))
    lifetime={str(key):{p.name:json.loads(p.read_text())['passed'] for p in folder.glob('*lifetime*.json')} for key,folder in folders.items()}
    passed=all(all(r[k] for k in ('producer_full_KV_exact','producer_hidden_exact','both_imports_exact','greedy_exact')) for r in rows)
    passed=passed and all(bool(v) and all(v.values()) for v in lifetime.values())
    write(ROOT/'analysis/BI0_exact_coalescing_N8_matched_instrumentation_v2.json',dict(passed=passed,rows=rows,lifetime=lifetime,evidence='CORRECTNESS_ONLY'))
    if not passed:
        write(ROOT/'state/coalescing_followup_state_hold.json',dict(time=time.time(),status='BLOCKED_CORRECTNESS',scope='No N32 performance after a failed matched-instrumentation state gate.'));return
    results=[]
    for enabled in (False,True):
        name=f'CoalescingAB_{int(enabled)}_Native_KV16_N32_q2.35_BI0_v2'
        value=serving(name,'Native','qualification32.json',32,2.35,2026112033,
            nixl_manual_coalesce=enabled,restore_official_UCX_init=True,phase='MATCHED_EXACT_DESCRIPTOR_COALESCING_INTERVENTION')
        setup=json.loads((ROOT/'raw'/name/'NIXL_host_setup_decode.json').read_text())['rows']
        window=json.loads((ROOT/'raw'/name/'window.json').read_text())
        selected=[r for r in setup if r['kind'] in ('make_transfer_host','submit_transfer_host','NIXL_native_telemetry') and
            window['start']<=r.get('start',r.get('time',0))<=window['end']]
        results.append(dict(name=name,enabled=enabled,metrics=value,transport_rows=selected))
        write(ROOT/'analysis/BI0_exact_coalescing_N32_screen_v2.json',dict(results=results,evidence='DIAGNOSTIC',
            scope='One independent start per arm, after matched synchronized N8 state gate. Exact KV byte mapping. '
            'No model/scheduler change. Natural outputs and batches remain explicit. No full200 primary adoption.'))
    write(ROOT/'state/exact_coalescing_followup_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/exact_coalescing_followup_hold.json',dict(time=time.time(),error=repr(error)));raise
