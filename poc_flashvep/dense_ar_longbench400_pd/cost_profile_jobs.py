"""Separate operator and transport hypotheses with matched serial controls."""
import json
from .common import ROOT
from .experiment import create
from .next_jobs import plan
from .geometry import matched_suffixes


def main():
    lanes=[[],[]]
    for bucket,cuts in ((0,(2,4)),(3,(4,))):
        for cut in cuts:
            for axis in ('Layer','Token'):
                lane=0 if axis=='Layer' else 1;pair=(2,3) if lane==0 else (4,5)
                extra={}
                if axis=='Token':
                    data=json.loads((ROOT/f'data/input{bucket}_discovery_exact16.json').read_text())
                    suffixes,audit=matched_suffixes([r['input_tokens'] for r in data],cut)
                    assert audit['exact_match']
                    extra=dict(suffix_by_request={'q'+str(i):s for i,s in enumerate(suffixes)},aggregate_work_audit=audit)
                for variant,role in [('Clean1',None),('P_Operators','prefill'),('D_Operators','decode'),('Clean2',None)]:
                    active=role is not None
                    lanes[lane].append(create(f'Cost_I{bucket}_{axis}{cut}_O8_{variant}_v2',axis,
                        f'input{bucket}_discovery_exact16.json',cut=cut,pair=pair,output=8,fixed=True,
                        controlled=True,serial=True,qps=20,seed=2026112211,
                        qualification_gate='C0_full_state_q4_gate.json',kernel_activity=active,
                        kernel_activity_roles=[role] if active else [],kernel_activity_seconds=7200,
                        operator_correlation=active,runtime_api_activity=active,
                        activity_copies=role=='prefill',
                        scope='Matched serial host-service and actual serving-operator diagnostic. '
                            'Each profile needs its own clean/profile/clean observer gate. Not an online TTFT speedup.',**extra))
    for index,jobs in enumerate(lanes):plan(f'operator_cost_lane{index}_v2',jobs)
    print('Frozen matched serial operator captures and bracketing controls.',flush=True)


if __name__=='__main__':main()
