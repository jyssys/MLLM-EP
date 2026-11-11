"""Select on N64. Confirm on new arrival seeds with balanced run order."""
from .common import *
from .prepare_serving import cell


RULE='Require exactness, absolute additional-work Decode gates, and no extra ITL threshold misses. Prefer the Native-relative 5% point gate. Then minimize TTFT p95. If only the relative gate fails, retain an explicitly negative-tail confirmation.'


def main():
    initialize();data=json.loads((ROOT/'analysis/serving_results.json').read_text())
    data=[r for r in data if r.get('complete') and r['count']==64]
    choices=[];jobs=[]
    for model,cap in sorted({(r['model'],r['cap']) for r in data}):
        rows=[r for r in data if r['model']==model and r['cap']==cap]
        zero=next(r for r in rows if r['width_arm']=='zero');selected=['zero']
        for family in ('loop','token'):
            candidates=[r for r in rows if r['width_arm'].startswith(family)]
            safe=[r for r in candidates if r['exact_output_work_lifetime_passed'] and r['Decode_SLO_screen'] and r['ITL_limit_misses']<=zero['ITL_limit_misses']]
            relative=[r for r in safe if r['ITL_p99_ms']<=1.05*zero['ITL_p99_ms'] and r['TPOT_p95_ms']<=1.05*zero['TPOT_p95_ms']]
            pool=relative or safe
            if not pool:
                choices.append(dict(model=model,cap=cap,family=family,status='NO_SAFE_STATIC_ARM_FOR_CONFIRMATION'))
                continue
            best=min(pool,key=lambda r:r['TTFT_p95_ms']);selected.append(best['width_arm'])
            choices.append(dict(model=model,cap=cap,family=family,arm=best['width_arm'],
                status='RELATIVE_POINT_GATE_PASS' if relative else 'NEGATIVE_TAIL_CONFIRMATION',
                selection_name=best['name'],selection_metrics={f:best[f] for f in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','ITL_limit_misses')}))
        for start,seed in enumerate((2026111111,2026111112,2026111113)):
            order=selected[start%len(selected):]+selected[:start%len(selected)]
            for arm in order:
                jobs.append(cell(model,arm,128,cap,seed,'HELD_OUT_ARRIVAL_CONFIRMATION_SELECTED_ON_N64','v2'))
    write(ROOT/'state/confirmation_selection.json',dict(rule=RULE,choices=choices,jobs=jobs,
        scope='The rule is defined before N64 serving results. Prompt IDs repeat the preserved pool. Only arrival seeds and process starts are new.'))
    print('CONFIRMATION_JOBS',len(jobs));print(json.dumps(choices))


if __name__=='__main__':main()
