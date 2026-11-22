"""Freeze capacity, selection and replay rules before clean C3 results."""
import hashlib
import json
import time
from .common import ROOT,write,digest

def main():
    path=ROOT/'configs/c3_clean_performance_contract_v2.json';assert not path.exists()
    discovery=json.loads((ROOT/'data/balanced400_v2_discovery.json').read_text())
    ordered=sorted(discovery,key=lambda r:hashlib.sha256(('capacity-v2:'+r['prompt_id']).encode()).hexdigest())
    pilot=[]
    for bucket in range(4):pilot.extend([r for r in ordered if r['input_bucket']==bucket][:16])
    pilot.sort(key=lambda r:hashlib.sha256(('capacity-order-v2:'+r['prompt_id']).encode()).hexdigest())
    assert len(pilot)==64;data=ROOT/'data/c3_capacity_discovery64_v2.json';write(data,pilot)
    contract=dict(time=time.time(),pilot_dataset=data.name,pilot_dataset_SHA256=digest(data),
        source='v2_native_decode_scoped_prompt_ops_v4.zip',
        pilot_QPS=[.7,1.4,2.8,4.0],pilot_methods=['NativeStrong','Agg'],
        pilot_rule='Use clean,unprofiled and already-qualified engines. Find the first offered QPS with achieved finite-cohort request rate below90% of offered, '
            'or with TTFT p95 above1.5 times the lowest-load value and post-arrival drain above25% of the arrival span. '
            'This is a finite-cohort knee estimate,not sustainable capacity. If no knee is observed,use the highest pilot point as a bracket,not a capacity claim.',
        grid_rule='Use the smaller observed Native/Agg knee. Multiply it by[0.20,0.35,0.50,0.65,0.80,1.00,1.20,1.50]. '
            'Round QPS to two decimals and deduplicate. Keep at least0.20 QPS. Freeze the common grid before candidate results.',
        discovery_dataset='balanced400_v2_discovery.json',
        discovery_dataset_SHA256=digest(ROOT/'data/balanced400_v2_discovery.json'),
        discovery_QPS_rule='Use the common grid point nearest the finite-cohort knee. The same trace,order and GPU pair apply to all arms within a block.',
        tuning_geometry='Layer cuts2/4/8/12. Strong compiled Stock Token cuts2/4/8/12 with uncapped coordinator credit0,and StockToken8 credit8. '
            'Hybrid(2,1/8),(4,1/8),(8,1/8),(4,1/4). Initial D admission cap2. NativeStrong,common-policy Native,Agg and whole-prefill deflection remain visible.',
        second_tuning_stage='After geometry discovery,qualify and measure the best Layer,Stock Token,Hybrid and deflection under the common prospective D-cap set0/1/2/4. '
            'Cap0 removes only the running-request cap. It retains installed decode-first compute,READ and registration deferral. '
            'It does not authorize active-decode insertion. Test credits8/16 for Layer and Hybrid,and0/8/16 for Stock Token,as budget permits. '
            'Every changed runtime signature requires a fresh state/quality composition proof. Unrun cases remain NOT_RUN.',
        selection='Within each family,first restrict to correctness and task-quality PASS. Prefer points with at least95% absolute decode attainment and '
            'ITL p99 and TPOT p95 point ratios at most1.05 against both Native references. Then minimize TTFT p95. '
            'Break ties by greater SLO-goodput,then the lexicographic configuration label. '
            'If no candidate passes this discovery point screen,retain the fastest qualified candidate as SAFETY_UNCONFIRMED. '
            'This screen is not the independent-start CI gate. Do not call a retained unsafe candidate a serving success.',
        confirmation='Freeze one configuration per family before heldout measurements. Use fresh independent starts and arrival seeds. '
            'Important points require at least3 paired starts. Primary CI uses the ratio of paired start means. No request-level performance bootstrap.',
        natural_output_confound='For every compared natural-EOS pair,report actual token totals,length distributions,cap hits,quality and exact-match counts. '
            'If total output tokens differ by more than5% from either comparator,flag OUTPUT_WORK_UNMATCHED. Do not attribute raw throughput or residency differences solely to scheduling.',
        matched_replay='At flagged important cells,run a separate diagnostic using the original compiled C1 per-request natural lengths for every arm. '
            'Use ignore_eos with exactly those lengths. Keep prompts,arrival and runtime unchanged. '
            'Label this CONTROLLED_REPLAY,not natural generation. Reference lengths are experimental output control,not an online selector feature. '
            'Do not replace or remove the natural-EOS result.',
        safety='Require paired95%CI upper<=1.05 for ITL p99 and TPOT p95 against NativeStrong and exact common-D-policy Native. '
            'Retain the frozen TTFT5000ms,every-ITL34ms and TPOT33ms metrics. Report maxITL and all-ITL request attainment. '
            'Relative safety and absolute attainment remain separate. No TTFT gain can cancel a decode failure.',
        safe_window='Physical kernel gaps are not executable windows. Do not run new active-decode insertion without a validated deadline,readiness,stream,credit and tail-cost predictor.',
        exposure='All400 requests had C1 reference exposure. Heldout192 is context-disjoint from discovery208. It is held out from C3 method tuning, '
            'not globally unseen after C0 fixed-output experiments. Report the additional old compiler-control overlap separately.')
    write(path,contract);print('C3_CLEAN_PERFORMANCE_PROTOCOL_FROZEN',digest(path),flush=True)

def amend_knee():
    assert not (ROOT/'jobs/c3_clean_capacity_pilots_v2.json').exists(),'Amend only before clean capacity pilots'
    path=ROOT/'configs/c3_capacity_knee_amendment_v2.json';assert not path.exists()
    write(path,dict(time=time.time(),parent_contract_SHA256=digest(ROOT/'configs/c3_clean_performance_contract_v2.json'),
        rule='Compare completed request rate against90% of the realized scheduled Poisson arrival rate,not90% of nominal offered QPS. '
            'Keep the TTFT/drain branch,grid multipliers,requests,seeds and metric definitions unchanged. '
            'Preserve offered QPS as the graph coordinate and report realized arrival separately.',
        reason='A finite Poisson trace can realize fewer arrivals per second than nominal offered QPS. '
            'The original rate branch could incorrectly label that sampling fluctuation as server saturation.',
        clean_capacity_results_seen=False,evidence='PROSPECTIVE_MEASUREMENT_CONTRACT_CORRECTION'))
    print('CAPACITY_KNEE_RULE_CORRECTED_BEFORE_ANY_CLEAN_PILOT',flush=True)

if __name__=='__main__':
    import sys
    amend_knee() if len(sys.argv)>1 and sys.argv[1]=='amend_knee' else main()
