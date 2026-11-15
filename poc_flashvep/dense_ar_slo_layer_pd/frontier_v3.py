"""Retune standard continuous replicas after exact generic corrections."""
from .common import *
from .configuration import base
from .frontier import run
from .analyze import metrics


def make(budget,slots,cap=16,rate=4.,placement='ROUND_ROBIN',count=64):
    name=f'frontier3_Colocated_B{budget}_S{slots}_O{cap}_q{rate}_{placement}_N{count}'
    return base(name,'Colocated',count=count,cap=cap,rate=rate,chunk=budget,token_budget=budget,
        D_slots=slots,placement=placement,verified_ByteLevel_priming=True,direct_phase_norm=True,
        phase='EXACT_GENERIC_RUNTIME_SLO_FRONTIER_DISCOVERY')


def main():
    assert json.loads((ROOT/'analysis/generic_runtime_full_state_gate.json').read_text())['passed']
    jobs=[make(b,16) for b in (128,256,384,512,768)]
    run(jobs)
    # Select a token budget from actual request-level objectives, not a
    # presumption that a small chunk or a large chunk protects Decode.
    values=[(c,metrics(c['name'])) for c in jobs]
    chosen=max(values,key=lambda x:x[1]['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput'])[0]
    budget=chosen['token_budget']
    run([make(budget,s,placement=p) for s,p in ((4,'ROUND_ROBIN'),(8,'ROUND_ROBIN'),
        (32,'ROUND_ROBIN'),(16,'LEAST_OUTSTANDING'))])
    run([make(b,32,rate=6.) for b in (8192,32768)])
    run([make(b,s,cap=64,rate=3.) for b,s in ((budget,16),(512,16),(8192,1))
        if not (b==512 and budget==512 and s==16)])
    write(ROOT/'state/frontier_v3_complete.json',dict(time=time.time(),selected_for_sequence_budget=budget,
        scope='Discovery only. Sustainable capacity and held-out superiority remain untested. The token-budget choice controls the next stage, not final qualification.'))


if __name__=='__main__':main()
