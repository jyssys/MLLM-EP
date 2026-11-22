"""Predeclare three warm-shape calibration triplets and retain every result."""
import json
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .next_jobs import plan


def main():
    capsule,sha=freeze('operator_API_warm_calibration_v3_20261121')
    lanes=[[],[],[]];tests=[]
    for case,(bucket,cut) in enumerate(((0,2),(0,4),(3,4))):
        first=(2,4,6)[case];dataset=f'input{bucket}_discovery_exact16.json'
        data=json.loads((ROOT/'data'/dataset).read_text());suffixes,work=matched_suffixes([r['input_tokens'] for r in data],cut)
        assert work['exact_match']
        for axis in ('Layer','Token'):
            for start in range(3):
                group=[]
                for label in ('Clean1','BothGPU_Operators','Clean2'):
                    enabled=label=='BothGPU_Operators'
                    extra=dict(suffix_by_request={'q'+str(i):s for i,s in enumerate(suffixes)},aggregate_work_audit=work) if axis=='Token' else {}
                    name=create(f'WarmCost_I{bucket}_{axis}{cut}_Start{start}_{label}_v3',axis,dataset,
                        cut=cut,pair=(first,first+1),qps=1,seed=2026112301+start,output=8,fixed=True,
                        controlled=True,serial=True,qualification_gate='C0_full_state_q4_gate.json',
                        source_capsule=capsule,source_capsule_SHA256=sha,warmup_all_prompts=True,
                        kernel_activity=enabled,kernel_activity_roles=['prefill','decode'],kernel_activity_seconds=7200,
                        operator_correlation=enabled,activity_copies=False,runtime_api_activity=enabled,
                        collector_namespace='correlated_collector_api_v2',
                        scope='Isolated warm-shape serving-API calibration. Warming uses benchmark prompt foreknowledge. '
                            'This is not online serving or a replacement Primary baseline. '
                            'Exactly three predeclared triplets are retained. No repeat-until-pass procedure.',**extra)
                    group.append(name)
                lanes[case].extend(group);tests.append(dict(input_bucket=bucket,cut=cut,axis=axis,start=start,jobs=group))
    write(ROOT/'configs/operator_repeat_observer_contract_v3.json',dict(tests=tests,
        evidence='FROZEN_ISOLATED_CALIBRATION_DESIGN',source_capsule_SHA256=sha,
        observer_rule='For each role,retain all three triplets. Bootstrap paired profile/mean-own-clean start blocks. '
            'Require the95percent interval to remain within[.95,1.05] for TTFTp95,meanTTFT,ITLp99,TPOTp95,and meanTBT. '
            'Require complete kernel attribution and zero dropped/invalid records. Report individual triplet failures too.',
        API_export='All already-collected runtime/driver API timestamps are exported in v2. v1 filters these out without GRAPH_TRACE. '
            'No API timing is reconstructed for old raw records.'))
    for lane,jobs in enumerate(lanes):plan(f'operator_repeat_warm_lane{lane}_v3',jobs)
    print('Frozen54 starts in18 observer triplets.',flush=True)


if __name__=='__main__':main()
