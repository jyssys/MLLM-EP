"""Freeze selections from viewed validation data, never fresh outcomes."""
from .common import *
from .confirmation import freeze

FIELDS=('D_admission_batch','D_token_budget','D_slots','max_reserved_requests','native_continuous_D_ingress',
    'defer_completion_log','token_budget','chunk','placement','static_layers','migration_fraction',
    'static_fraction_axis','static_request_block_size','cost_model','revised_revision',
    'maximum_predicted_D_wait_ms','maximum_queue_groups','safety_margin_ms','safety_margin_by_layers_ms',
    'exclude_negotiation_from_D_occupancy','queue_benefit_weight','decode_prediction','decode_prediction_margin_ms')


def settings(row):
    cfg=json.loads((ROOT/'configs'/(row['name']+'.json')).read_text())
    return {k:cfg[k] for k in FIELDS if k in cfg}


def main():
    assert not (ROOT/'configs/heldout_selection.json').exists()
    rows=json.loads((ROOT/'analysis/final_viewed_validation.json').read_text())['results']
    assert all(r['output']['passed'] for r in rows)
    chosen={};audit=[]
    for cap in (16,64,200):
        pool=[r for r in rows if r['metrics']['output_cap']==cap]
        native=next(r for r in pool if r['name']==f'validation_Native_cohort_O{cap}_N64_v1')
        reference=native['metrics']
        def safe(row):
            m=row['metrics']
            return (m['ITL_p99_ms']<=1.05*reference['ITL_p99_ms'] and
                m['TPOT_p95_ms']<=1.05*reference['TPOT_p95_ms'] and
                m['maximum_ITL_ms']<=max(reference['maximum_ITL_ms'],m['SLO_regimes']['Balanced']['thresholds']['ITL_ms']) and
                m['SLO_regimes']['Balanced']['percentile_Decode_absolute_pass'])
        colocated=[r for r in pool if '_Colocated_' in r['name']]
        colocated_best=max(colocated,key=lambda r:(r['metrics']['SLO_regimes']['Balanced']['Request_SLO_goodput'],
            r['metrics']['SLO_regimes']['Balanced']['Decode_attainment'],r['metrics']['output_tokens_s'],-r['metrics']['TTFT_p95_ms']))
        static=[r for r in pool if '_Static' in r['name'] and safe(r)]
        static_best=min([native,*static],key=lambda r:r['metrics']['TTFT_p95_ms'])
        revised=[r for r in pool if '_Revised_' in r['name']]
        feasible=[r for r in revised if safe(r)]
        if feasible:revised_best=min(feasible,key=lambda r:r['metrics']['TTFT_p95_ms'])
        else:
            revised_best=min(revised,key=lambda r:(max(r['metrics']['ITL_p99_ms']/reference['ITL_p99_ms'],
                r['metrics']['TPOT_p95_ms']/reference['TPOT_p95_ms']),r['metrics']['TTFT_p95_ms']))
        selected=dict(Native=settings(native),Colocated=settings(colocated_best),Static=settings(static_best),Revised=settings(revised_best))
        if static_best is native:selected['Static']['_base_arm']='Native'
        elif static_best['migrated_requests']==static_best['metrics']['count']:
            # The ingress option has no measured Native traffic in this fixed
            # all-migrated rule. Keep any future fallback Decode-isolated.
            selected['Static']['native_continuous_D_ingress']=False
        chosen[str(cap)]=selected
        audit.append(dict(cap=cap,native_reference=native['name'],
            selected_names=dict(Native=native['name'],Colocated=colocated_best['name'],Static=static_best['name'],Revised=revised_best['name']),
            candidate_safety={r['name']:safe(r) for r in pool},
            point_only=True,scope='The Decode-isolated Native remains the protection reference. Native Continuous cannot enlarge the non-inferiority target. Static includes Native Last0.'))
    selection=dict(settings_by_cap=chosen,native_settings=chosen['200']['Native'],selection_audit=audit,
        rules='Native: fixed Decode isolation. Colocated: maximum Balanced goodput, then Decode attainment and generation throughput. Static: minimum TTFT among point-safe fixed rules and Native Last0. Revised: minimum TTFT among point-safe revised policies, otherwise least Decode ratio violation. Confirmation remains mandatory.',
        validation_SHA256=digest(ROOT/'analysis/final_viewed_validation.json'))
    freeze(selection);print('SELECTION_FROZEN',[(r['cap'],r['selected_names']) for r in audit],flush=True)


if __name__=='__main__':main()
