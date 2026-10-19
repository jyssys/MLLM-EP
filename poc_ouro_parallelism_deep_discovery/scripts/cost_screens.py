"""Charge candidate mechanisms. Estimates are not observed serving paths."""
import argparse
import csv
import statistics
from poc_ouro_parallelism_deep_discovery.contract import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--native-summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    native=list(csv.DictReader(args.native_summary.open()))
    correction=[]
    for row in native:
        if row['P']=='1':continue
        reference=next(r for r in native if (r['model'],r['R'],r['P'],r['context'])==(row['model'],row['R'],'1',row['context']))
        # A TPOT contains more than core work. This proportional screen is a
        # calibration approximation, not a hardware lower bound.
        F1=float(reference['median_TPOT_ms'])/4
        matched=float(row['median_TPOT_ms']);P=int(row['P'])
        for warmup,rounds in [(0,2),(0,3),(0,4),(1,1),(1,2),(1,3)]:
            invocations=warmup+4*rounds
            charged_rounds=warmup+((4+P-1)//P)*rounds
            optimistic=charged_rounds*F1
            active=[max(0,4-warmup-i) for i in range(rounds)]
            prefix_rounds=warmup+sum((n+P-1)//P for n in active)
            correction.append(dict(model=row['model'],P=P,context=int(row['context']),
                exact_warmup=warmup,correction_rounds=rounds,expensive_core_evaluations=invocations,
                idealized_serial_rounds=charged_rounds,native_TPOT_ms=matched,
                approximate_core_round_ms=F1,optimistic_core_span_ms=optimistic,
                maximum_extra_budget_ms=matched-optimistic,
                modeled_same_P_speedup_before_extra_cost=matched/optimistic,
                exact_prefix_skipping_alternative=dict(active_depths_per_round=active,
                    ideal_serial_rounds=prefix_rounds,
                    expensive_core_evaluations=warmup+sum(active),
                    optimistic_core_span_ms=prefix_rounds*F1,
                    modeled_same_P_speedup_before_extra_cost=matched/(prefix_rounds*F1),
                    status='Analytic schedule refinement, not the executed full-Jacobi implementation. It assumes exact-prefix detection and preserved matching KV are free.'),
                exclusions=['State exchange','Validation','Current-token KV repair','Prediction/correction arithmetic','Contention','Extra allocation'],
                evidence_class='MODEL_ESTIMATE',
                limitation='Native TPOT/4 is only a coarse core-time approximation. It is not a measured recurrent-pass timer.',
                quality='Only warm0/K4 and warm1/K3 recover exact states in the bounded current-token probes. Other schedules remain approximate and unvalidated.'))
    remote=[]
    for path in sorted((ROOT/'outputs').glob('components_*_graph_*/components.json')):
        content=read(path)
        for item in content['results']:
            local=item['local']['median_critical_stream_ms']
            tp=item['TP']['median_critical_stream_ms']
            pooled=item['remote_round_trip']['median_critical_stream_ms']
            L=next(v['layers'] for v in MODELS.values() if v['id']==item['model'])
            remote.append(dict(model=item['model'],P=item['P'],rows=item['rows'],
                local_MLP_ms=local,TP_MLP_ms=tp,remote_MLP_ms=pooled,
                remote_vs_TP_component_speedup=tp/pooled,
                hypothetical_all_R4_MLP_span_increment_ms=(pooled-tp)*L*4,
                payload_one_way=item['activation_bytes_one_way'],
                input=str(path.relative_to(ROOT)),evidence_class='MODEL_ESTIMATE',
                measured_input_class='GPU_MICROBENCH',
                scope='Serialize matched median component costs for a screen. This is not measured full-model E2E or a sum of aligned rank timestamps.',
                excluded='Different pooling batch sizes, other ready requests, queuing, whole-stack reuse distance, attention contention'))
    compression=[]
    transport=ROOT/'outputs/communication_P2_v1/communication.json'
    if transport.exists():
        records=read(transport)['records']
        for amount in [4096,131072]:
            link=next(r for r in records if r['primitive']=='broadcast' and r['payload_bytes']==amount)
            transform=next(r for r in records if r['primitive']=='XOR_byteplane_roundtrip_no_compression' and r['payload_bytes']==amount)
            raw=link['timing']['median_ms'];prep=transform['timing']['median_ms']
            compression.append(dict(payload_bytes=amount,raw_broadcast_ms=raw,
                measured_unfused_transform_ms=prep,transform_over_raw_ms=prep/raw,
                necessary_transport_saving_ms=prep,
                evidence_class='MODEL_ESTIMATE',
                excluded='Entropy encoding/decoding, metadata, base-version exchange, fused implementation, real branch runtime',
                limitation='Separate primitive medians do not form an observed pipeline. Transform is only the measured unfused round trip.'))
    write(args.output,dict(correction=correction,remote_pool=remote,lossless=compression,
        native_summary=str(args.native_summary),native_summary_sha256=sha(args.native_summary),
        evidence_class='MODEL_ESTIMATE',candidate_live_E2E='NOT_RUN',
        inference_rule='No modeled value is promoted to achieved speedup. Negative screens apply only to the named transformations.'))
    print('COST_SCREENS',len(correction),len(remote),len(compression))

if __name__=='__main__':main()
