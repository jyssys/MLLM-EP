"""Test a Host-output hypothesis with the exact same generated token IDs."""
import json
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics,compare,records,percentile
from .resume_campaign import restore,record


def main():
    ref='graph_qwen8_L8000_O16_N32_native_event_diagnostic_v29'
    c=json.loads((ROOT/'configs'/(ref+'.json')).read_text())
    c.update(name='graph_qwen8_L8000_O16_N32_native_no_detokenization_diagnostic_v30',
        ar_detokenize=False,phase='HOST_DETOKENIZATION_COUNTERFACTUAL_DIAGNOSTIC',
        scope='Only Host text detokenization changes. Original BF16 model, exact Attention, output token IDs, fixed cap, and SLO remain unchanged. This engine-API harness delivers token IDs, not HTTP text.')
    name=save(c)
    try:
        if not execute(name,restore=False):return
        gate=compare(ref,name);rows=records(ROOT/'raw'/name/'requests.jsonl');first=[];steady=[]
        for r in rows:
            for i,(a,b) in enumerate(zip(r['token_events'],r['token_events'][1:])):
                (first if i==0 else steady).append(1000*(b['worker_time']-a['worker_time'])/(b['count']-a['count']))
        result=dict(name=name,reference=ref,output_gate=gate,metrics=metrics(name),reference_metrics=metrics(ref),
            first_Worker_ITL_p50_ms=percentile(first,.5),first_Worker_ITL_p99_ms=percentile(first,.99),
            steady_Worker_ITL_p99_ms=percentile(steady,.99),
            evidence='DIAGNOSTIC_HOST_OUTPUT_COUNTERFACTUAL',
            scope='A change in first transition does not identify exclusive CPU time without additional instrumentation. A successful token-only control is not a text-serving speedup.')
        write(ROOT/'analysis/detokenization_counterfactual.json',result)
        record('detokenization_diagnostic',dict(name=name,output_gate=gate,first_ITL_p99_ms=result['first_Worker_ITL_p99_ms']))
        print(json.dumps(result),flush=True)
    finally:restore()


if __name__=='__main__':main()
