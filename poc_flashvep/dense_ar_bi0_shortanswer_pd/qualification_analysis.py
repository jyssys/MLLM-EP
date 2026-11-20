"""Distinguish exact transport from legitimate BI0 execution-geometry changes."""
import json
from .common import ROOT,write
from .quality import quality_gate,GOLD


def main():
    names={label:f'qual_{label}_ShortAnswer_N8_BI0_state_v1' for label in ('Standard','Native','Agg')}
    states={}
    for label,name in names.items():
        folder=ROOT/'raw'/name
        states[label]={file.stem.removeprefix('prompt_state_'):json.loads(file.read_text())['requests']
                       for file in folder.glob('prompt_state_*.json')}
    teacher=states['Standard']['single'];rows=[]
    for rid,source in teacher.items():
        if not rid.startswith('q'):
            continue
        producer=states['Native']['prefill'][rid]
        destination=states['Native']['decode'][rid]
        rows.append(dict(request_id=rid,
            Native_P_KV_equals_same_geometry_Standard=producer['KV']==source['KV'],
            Native_P_hidden_equals_same_geometry_Standard=producer['final_hidden_SHA256']==source['final_hidden_SHA256'],
            imported_full_KV_equals_Native_P=destination['KV']==producer['KV'],
            imported_full_KV_equals_Standard=destination['KV']==source['KV'],
            D_first_query=destination['first_observed_query'],expected_D_first_query=producer['original_input_tokens']-1))
    quality={label:quality_gate(name) for label,name in names.items()}
    passed=len(rows)==8 and all(r['Native_P_KV_equals_same_geometry_Standard'] and
        r['Native_P_hidden_equals_same_geometry_Standard'] and r['imported_full_KV_equals_Native_P']
        and r['D_first_query']==r['expected_D_first_query'] for r in rows)
    passed=passed and quality['Standard']['strict_tokens_matched']==8
    write(ROOT/'analysis/BI0_N8_official_state_transport_gate.json',dict(passed=passed,rows=rows,
        quality={label:{k:value[k] for k in ('strict_tokens_matched','point_passed','CI_passed','tasks','lifetime')} for label,value in quality.items()},
        evidence='CORRECTNESS_ONLY',
        scope='Native P uses the same serial prompt geometry as the Standard reference. Imported valid full KV is checked before D recomputes its last prompt query. Cross-geometry output mismatches are reported, not overwritten. N8 does not certify full200 task noninferiority.'))
    return passed


if __name__=='__main__':
    print('BI0_N8_STRUCTURE',main())
