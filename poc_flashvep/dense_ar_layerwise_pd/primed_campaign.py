"""Keep text output and normalize the generic output lifecycle first."""
import json,re
from .common import *
from .prepare import save
from .run import execute
from .analyze import metrics,compare
from .graph_feasibility import gate
from .state_qualification import events
from .resume_campaign import restore,record


def main():
    assert json.loads((ROOT/'analysis/decoder_priming_CPU_gate.json').read_text())['passed']
    results=[]
    try:
        for count in (8,32):
            for label in ('native','c28'):
                c=json.loads((ROOT/'configs'/f'graph_qwen8_L8000_O16_N{count}_{label}_qual_v28.json').read_text())
                c.update(name=f'primed_graph_qwen8_L8000_O16_N{count}_{label}_qual_v31',
                    prime_detokenizer=True,verify_primed_text=True,async_ingress=True,
                    phase='EXACT_TEXT_PRIMING_AND_ASYNC_INGRESS_QUALIFICATION')
                name=save(c)
                if not execute(name,restore=False):return
                passed=gate(name,count,label!='native')
                parity={}
                for role in ('prefill','decode'):
                    rows=[r for r in events(name,role) if r['event']=='PRIMED_TEXT_PARITY' and
                        re.fullmatch(r'q\d+',r.get('external_request_id') or '')]
                    expected=count if role=='prefill' else count*(15 if label=='native' else 16)
                    parity[role]=dict(count=len(rows),expected=expected,passed=len(rows)==expected and all(r['passed'] for r in rows))
                passed=passed and all(r['passed'] for r in parity.values())
                row=dict(name=name,passed=passed,text_parity=parity)
                results.append(row);write(ROOT/'state/primed_campaign_progress_v31.json',dict(results=results))
                print('PRIMED_GATE',name,passed,parity,flush=True)
                if not passed:return
        arms=[('nativeA','native'),('c28','c28'),('colocated','colocated'),('nativeB','native')]
        for label,template in arms:
            c=json.loads((ROOT/'configs'/f'graph_serving_qwen8_L8000_O16_N64_s3_{template}_v28.json').read_text())
            c.update(name=f'primed_graph_serving_qwen8_L8000_O16_N64_s3_{label}_v31',
                prime_detokenizer=True,verify_primed_text=False,async_ingress=True,
                hidden_source_backpressure=template=='c28',phase='PRIMED_GENERIC_RUNTIME_STATIC_SELECTION',
                scope='Text detokenization remains enabled. Exact decoder priming and asynchronous RPC apply to every arm. No GPU arithmetic, output cap, arrival trace, KV budget, or SLO changes.')
            name=save(c)
            if not execute(name,restore=False):return
            ref='graph_serving_qwen8_L8000_O16_N64_s3_native_v28'
            row=dict(name=name,metrics=metrics(name),output_gate=compare(ref,name))
            results.append(row);write(ROOT/'state/primed_campaign_progress_v31.json',dict(results=results))
            print('PRIMED_PILOT',json.dumps(row),flush=True)
            if not row['output_gate']['passed']:return
    finally:restore()


if __name__=='__main__':main()
