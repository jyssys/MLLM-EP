"""Recompute primary metrics without replacing stored measurement files."""
import json
from unittest.mock import patch
from .common import ROOT,write
from . import metrics


FIELDS=('n','output_tokens','TTFT_mean_ms','TTFT_p95_ms','ITL_p99_ms','mean_TBT_ms',
        'TPOT_p95_ms','generated_output_tokens_per_second','completed_requests_per_second',
        'Joint_SLO_attainment','SLO_qualified_output_tokens_per_second')


def main():
    patterns=('Main_*_BI0_v3_metrics.json','ControlledDiscovery_*_BI0_v3_metrics.json',
        'Confirmation_*_BI0_v2_metrics.json','FixedQPS_*_BI0_v1_metrics.json',
        'RootCut12_*_BI0_v1_metrics.json','LowQPSControl_*_BI0_v1_metrics.json','EqualOutputControl_*_BI0_v1_metrics.json')
    rows=[]
    for path in sorted({p for pattern in patterns for p in (ROOT/'analysis').glob(pattern)}):
        original=json.loads(path.read_text());name=original['name']
        with patch.object(metrics,'write',lambda *args,**kwargs:None):fresh=metrics.summarize(name)
        differences={field:abs(original[field]-fresh[field]) for field in FIELDS
            if original[field] is not None and fresh[field] is not None}
        passed=all(value<=1e-6 for value in differences.values())
        passed &= all((original[field] is None)==(fresh[field] is None) for field in FIELDS)
        rows.append(dict(name=name,passed=passed,absolute_differences=differences,
            actual_interval_count=fresh['ITL_interval_count'],expected_interval_count=fresh['output_tokens']-fresh['n'],
            first_event_multi_token_requests=fresh['first_event_multi_token_requests']))
    result=dict(rows=rows,all_passed=all(r['passed'] for r in rows),
        evidence='TRACE_ANALYSIS_METRIC_REPRODUCTION',scope='Recompute raw client-event metrics with the final parser. '
        'Stored metrics and raw files remain unchanged. Absolute numeric tolerance is1e-6 in each declared unit. '
        'This does not add server starts or qualify task accuracy, GPU timing or sustainable capacity.')
    write(ROOT/'analysis/BI0_primary_metric_reproduction_audit_v1.json',result)
    assert result['all_passed'],'Do not publish inconsistent primary metrics'
    return result


if __name__=='__main__':main()
