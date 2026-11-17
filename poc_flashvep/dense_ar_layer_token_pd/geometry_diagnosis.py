"""Separate coordinator cohort limits from actual GPU batch observations."""
import collections
import json
import statistics
from .common import ROOT,write
from .analysis import records


def main():
    names=('adaptive_source_flow_H2_N64_O64_q4.8_v1',
        'flow8_H2_N64_O64_q4.8_v1','macro8_cal_Last4_N64_O64_q4.8_v1')
    values=[];configs=[]
    for name in names:
        folder=ROOT/'raw'/name
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());configs.append(cfg)
        result=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
        requests={'q'+str(r['sample_id']):r for r in records(folder/'requests.jsonl')}
        cohorts=collections.defaultdict(list)
        for r in records(folder/'D_host_admission.jsonl'):
            cohorts[r['cohort']].append(r)
        samples=[]
        for cohort,rows in cohorts.items():
            members=[r['request_id'] for r in rows]
            samples.append(dict(cohort=cohort,requests=members,count=len(members),
                admit_to_last_output_ms=1000*(max(requests[r]['finish'] for r in members)-min(r['time'] for r in rows)),
                first_to_last_output_ms=1000*(max(requests[r]['finish'] for r in members)-min(requests[r]['first_token'] for r in members))))
        values.append(dict(name=name,source_macro_limit=cfg['P_group_size'],P_slots=cfg['P_slots'],
            destination_admission_limit=cfg['D_admission_batch'],
            cohort_count=len(samples),actual_cohort_sizes=dict(collections.Counter(r['count'] for r in samples)),
            mean_first_to_last_output_ms=statistics.mean(r['first_to_last_output_ms'] for r in samples),
            samples=samples,actual_decode_GPU_shapes=json.loads((folder/'measured_geometry_decode.json').read_text()),
            raw_output_tokens_s=result['output_tokens_s'],TTFT_p95_ms=result['TTFT_p95_ms']))
    same={key:all(cfg[key]==configs[0][key] for cfg in configs) for key in
        ('dataset','arrival_offsets','request_caps','gpus')}
    write(ROOT/'analysis/longer_output_cohort_geometry_diagnosis.json',dict(
        comparisons=values,same_workload_checks=same,evidence='TRACE_ANALYSIS',
        structural_limit='A serialized whole-cohort guard cannot accept a new request before the old cohort retires. A cohort limit4 serves at most4 requests per such residency. A source macro8 can still execute as two P batches4.',
        scope='Actual cohort membership and client delivery spans are measured host boundaries. They are not exclusive GPU time or a physical-capacity ceiling. Cost model, cut policy and geometry also differ, so this is not a geometry-only causal ablation. This implementation constraint does not prove an absence of spatial GPU slack or a fundamental limitation of dynamic layer splitting.'))
    print('LONG_OUTPUT_COHORT_GEOMETRY_DIAGNOSIS',same,flush=True)


if __name__=='__main__':
    main()
