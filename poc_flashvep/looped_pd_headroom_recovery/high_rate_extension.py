"""Complete the predeclared Ouro 4/5-QPS pressure checks after safe warm-up."""
import copy
from .common import *
from .prepare import put
from .initial_campaign import run
from .primary_campaign import subset_parity
from .async_campaign import integrity


def main():
    while True:
        p=ROOT/'state/warm_campaign.json'
        if p.exists() and json.loads(p.read_text()).get('stage')=='WARM_EPOCH_COMPLETE_REVIEW_REQUIRED':break
        time.sleep(5)
    from .warm_campaign import make
    item=next(r for r in make()['models'] if r['model']=='ouro');ledger=[]
    gate=ROOT/'raw'/item['byte_gate']/'recovery_gate.json'
    if gate.exists() and json.loads(gate.read_text()).get('passed'):
        sources=[r for r in item['pilots'] if '_q3p0_' in r]+[r for r in item['DP_pilots'] if '_q3p0_' in r]
        for source in sources:
            for rate in (4.,5.):
                cfg=json.loads((ROOT/'configs'/(source+'.json')).read_text())
                if cfg['arm'].startswith('DP'):
                    gate_name=next(g for g in item['DP_gates'] if json.loads((ROOT/'configs'/(g+'.json')).read_text())['chunk']==cfg['chunk'])
                    proof=ROOT/'raw'/gate_name/'recovery_gate.json'
                    if not proof.exists() or not json.loads(proof.read_text()).get('passed'):
                        ledger.append(dict(name=source,status='NOT_RUN_NO_DP2_WORK_GATE'));continue
                cfg.update(name=source.replace('_q3p0_',f'_q{str(rate).replace(".","p")}_'))
                name=put(cfg,24,rate)
                write(ROOT/'state/high_rate_extension.json',dict(stage='BOUNDED_4_5_QPS_CHECK',name=name,time=time.time()))
                okay=run(name) and subset_parity(item['output_gate'],name) and integrity(name)
                ledger.append(dict(name=name,passed=okay));write(ROOT/'state/high_rate_results.json',ledger)
                if not okay:break
    write(ROOT/'state/high_rate_extension.json',dict(stage='HIGH_RATE_COMPLETE_REVIEW_REQUIRED',results=ledger,time=time.time(),
        scope='Predeclared 4/5 QPS, N24 bounded overload checks. Same original outputs and native safety credits. A long drain or growing queue is not a sustainable capacity result. No optional 7/10 QPS stress.'))
    restore_if_idle()


if __name__=='__main__':main()
