"""Check executed GPU geometry and resample matched independent starts."""
import json
from .common import ROOT,write
from .analysis import metrics
from .statistics_report import KEYS,ratio_interval


def geometry(name,role):
    return json.loads((ROOT/'raw'/name/f'measured_geometry_{role}.json').read_text())['counts']


def summarize(entries,target):
    pairs=[]
    for row in entries:
        a,b=row['name'],row['source']
        checks=dict(row['gate']['checks'])
        checks.update({role+'_actual_GPU_geometry_histogram':geometry(a,role)==geometry(b,role)
                       for role in ('prefill','decode')})
        pairs.append(dict(control=a,dynamic=b,checks=checks,passed=all(checks.values()),
            control_metrics=metrics(a),dynamic_metrics=metrics(b)))
    valid=[row for row in pairs if row['passed']]
    ratios={key:ratio_interval([r['control_metrics'][key] for r in valid],
        [r['dynamic_metrics'][key] for r in valid]) for key in KEYS
        if valid and key in valid[0]['control_metrics']}
    regimes={regime:{key:ratio_interval(
        [r['control_metrics']['SLO_regimes'][regime][key] for r in valid],
        [r['dynamic_metrics']['SLO_regimes'][regime][key] for r in valid])
        for key in ('Joint_attainment','Output_token_SLO_goodput')}
        for regime in ('TTFT_priority','Balanced','Decode_priority')} if valid else {}
    result=dict(
        pairs=pairs,qualified_pairs=len(valid),metrics=ratios,SLO_regimes=regimes,
        ratio_direction='Online Dynamic / retrospective matched histogram control',
        evidence='TRACE_ANALYSIS',performance_evidence='MEASURED',
        scope='Match actual source/destination request groups, cut histogram, completed work and role-specific executed GPU shape histograms. Placement in time intentionally differs. The control uses aggregate future cut counts, not request latency or future completion. It is a retrospective diagnostic, not a deployable static baseline. Bootstrap unit is the paired independent start. Matching aggregate shape counts does not match actual temporal overlap, cache state, retention duration or exclusive kernel time.')
    write(target,result)
    print('MATCHED_ACTUAL_GEOMETRY_PAIRS',len(valid),'OF',len(pairs),flush=True)
    return result


def main():
    source=ROOT/'analysis/heldout_exact_histogram_geometry_controls.json'
    entries=json.loads(source.read_text()) if source.exists() else []
    summarize(entries,ROOT/'analysis/heldout_policy_matched_actual_geometry_statistics.json')


if __name__=='__main__':
    main()
