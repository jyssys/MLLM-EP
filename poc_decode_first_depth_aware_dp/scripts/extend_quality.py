"""Extend Stage 0 from 16 to 32 frozen questions per domain. Do not tune the gate."""
import datetime
from poc_decode_first_depth_aware_dp.contract import *


def main():
    parent=ROOT/'artifacts/quality_protocol_v2.json'
    target=ROOT/'artifacts/quality_protocol_v3_pilot32.json'
    assert not target.exists()
    old=read(parent)
    summary=read(ROOT/'artifacts/quality_pilot_analysis_v1/quality_summary.json')
    assert summary['verdict']=='PILOT_UNCERTAIN_REQUIRES_FULL_QUALITY'
    # The immutable metadata order predates all model outputs.
    old_ids={r['id'] for r in old['rows'] if r['split']=='pilot'}
    selected=set()
    for domain in ['gsm8k','math500']:
        selected.update(r['id'] for r in [r for r in old['rows'] if r['dataset']==domain][:32])
    assert old_ids<=selected
    rows=[dict(r,split='pilot' if r['id'] in selected else 'heldout') for r in old['rows']]
    write(target,dict(old,rows=rows,pilot_per_domain=32,parent_protocol_sha256=sha(parent),
        before_any_inference=False,extension_before_its_inference=True,
        extension_ids=sorted(selected-old_ids),extension_frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selection='Next metadata-ordered 16 questions per domain. No selection from outputs.',
        sequential_analysis='Planned bounded 32-question pilot continuation. Nominal CIs, not anytime-valid inference.',
        gate_unchanged=True,accuracy_guardrails_unchanged=True,
        extension_reason='The first 16 questions per domain do not establish quality equivalence. Stage 0 only.'))
    print('PILOT32_FROZEN',len(selected),'paired questions; new questions',len(selected-old_ids))


if __name__=='__main__':main()
