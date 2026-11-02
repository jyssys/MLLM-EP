"""Test a supported Decode slot limit against observed four-row graph overflow."""
import copy
from .common import *
from .prepare import put
from .initial_campaign import run,gate
from .primary_campaign import subset_parity
from .async_campaign import integrity


def main():
    while True:
        p=ROOT/'state/warm_campaign.json'
        if p.exists() and json.loads(p.read_text()).get('stage')=='WARM_EPOCH_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    from .warm_campaign import make
    item=next(r for r in make()['models'] if r['model']=='huginn');ledger=[]
    base=json.loads((ROOT/'configs'/(item['byte_gate']+'.json')).read_text())
    proof=ROOT/'raw'/item['byte_gate']/'recovery_gate.json'
    accepted=proof.exists() and json.loads(proof.read_text()).get('passed')
    if accepted:
        for n,heavy in ((2,True),(8,False)):
            c=copy.deepcopy(base)
            c.update(name=f'warm_huginn_graph_limit_Dslots4_gate{n}',D_slots=4,phase='correctness_graph_limit_control',
                verify_transfer=heavy,state_probe=heavy)
            name=put(c,n);okay=run(name) and subset_parity(item['output_gate'],name) and integrity(name)
            if okay:
                configure()
                from ..looped_pd_atomic_prefill import evidence
                evidence.ROOT=ROOT
                check=dict(work=evidence.full_work(name))
                if heavy:check['transfer']=evidence.transfer(name)
                check['passed']=all(v['passed'] for v in check.values())
                write(ROOT/'raw'/name/'recovery_gate.json',check);okay=check['passed']
            ledger.append(dict(name=name,passed=okay));write(ROOT/'state/graph_safe_decode_results.json',ledger)
            if not okay:accepted=False;break
    if accepted:
        for chunk in (2560,2048):
            for rate in (.6,1.,1.2):
                c=copy.deepcopy(base)
                c.update(name=f'warm_huginn_Dslots4_c{chunk}_q{str(rate).replace(".","p")}_pilot24',
                    D_slots=4,chunk=chunk,phase='qps_pilot',state_probe=False,verify_transfer=False)
                name=put(c,24,rate)
                write(ROOT/'state/graph_safe_decode_control.json',dict(stage='D_GRAPH_LIMIT_CONTROL',name=name,time=time.time()))
                okay=run(name) and subset_parity(item['output_gate'],name) and integrity(name)
                ledger.append(dict(name=name,passed=okay));write(ROOT/'state/graph_safe_decode_results.json',ledger)
                if not okay:break
    write(ROOT/'state/graph_safe_decode_control.json',dict(stage='GRAPH_LIMIT_CONTROL_COMPLETE_REVIEW_REQUIRED',results=ledger,time=time.time(),
        hypothesis='The existing native graph supports <=4 query rows. The previous D-slot limit allows 5-row eager fallbacks. Limiting active D compute to four rows should remove that path without changing precision, depth, or graph implementation.',
        alternative_hypotheses=['KV admission can remain binding even if the tail improves.','CPU submission or other scheduling variation can retain tails despite removing graph overflow.'],
        scope='Generic supported scheduler tuning. Not a Looped-specific method. Native reservation and ACK checks remain enabled.'))
    restore_if_idle()


if __name__=='__main__':main()
