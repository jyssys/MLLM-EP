"""Analytical capacity and dependency filters. No invented architecture speedup."""
import csv
from poc_dllm_multigpu_e2e_atlas.contract import *
from poc_dllm_multigpu_e2e_atlas.scripts.parse_timeline import table


def main():
    layout=json.loads((ROOT/'results/raw/native_p16384_v1/native_layout.rank0.json').read_text())
    weights=layout['weights']
    def sharded(name):
        return any(p in name for p in ['query_key_value.weight','attention.dense.weight',
            'experts.w13_weight','experts.w2_weight','gate_up_proj.weight','down_proj.weight'])
    shard=sum(w['bytes'] for w in weights if sharded(w['name']))
    replicated=sum(w['bytes'] for w in weights if not sharded(w['name']))
    full=4*shard+replicated
    data=[]
    for name,tp,pp,dp,kind in [('TP4',4,1,1,'MEASURED'),('TP2xDP2',2,1,2,'PREDICTION_ONLY'),
        ('TP1xDP4',1,1,4,'PREDICTION_ONLY'),('PP2xTP2',2,2,1,'ARCHITECTURE_NOT_AVAILABLE'),
        ('PP4xTP1',1,4,1,'ARCHITECTURE_NOT_AVAILABLE')]:
        weight_proxy=(4*shard/tp/pp+replicated)
        data.append(dict(layout=name,TP=tp,PP=pp,DP=dp,physical_GPUs='4/5/6/7',GPU_count=tp*pp*dp,
            status=kind,weight_GiB_per_rank_proxy=weight_proxy/2**30,
            capacity='Weights fit theoretical80GB. Candidate graph/KV peak NOT_RUN.' if name!='TP4' else 'Physical peak measured.',
            B1_latency='New measured TP4 baseline' if name=='TP4' else 'NOT_MEASURED. Lower-TP compute/KV cost cannot be assumed equal.',
            B2_B4_throughput='NOT_RUN. Independent requests only. No B1 latency inference.',
            single_wave_PP_util_B1=1/pp,single_wave_PP_util_B2=2/(2+pp-1),single_wave_PP_util_B4=4/(4+pp-1),
            bubble_scope='Ideal balanced one-wave PP bound, not sustained iterative throughput prediction',
            communication='Two TP all-reduces/layer plus native sampling broadcast' if tp>1 else 'PP boundary transfers' if pp>1 else 'No inter-rank collective within one replica',
            realizability='Native B1 validated' if name=='TP4' else
                'TP-only code supports degree changes, but independent replica launch and graph memory need validation' if pp==1 else
                'Model has PPProxyTensors. Native diffusion runner lacks validated stage transport/feedback and cache ownership.',
            novelty='Generic TP/DP/PP. FastPP and block-serving prior constrain novelty.',
            expected_improvement='UNRESOLVED without lower-TP operator calibration and real concurrency. No ideal scaling claim.'))
    table(ROOT/'results/architecture_screening.csv',data)
    baseline=json.loads((ROOT/'results/bootstrap_cis.json').read_text())['prefixes']
    valid=[p for p in baseline if baseline[p]['observer_valid']]
    write(ROOT/'results/architecture_decision.json',dict(
        actual_TP4_parameter_bytes=layout['parameter_bytes'],checkpoint_tensor_bytes=32511286784,
        derived_full_loaded_weight_bytes_proxy=full,sharded_parameter_bytes_TP4=shard,
        replicated_parameter_bytes_TP4=replicated,
        caveat='PP estimate does not prove stage balance or workspace fit. Unconditional embeddings and last-stage heads need exact construction.',
        context_critical_evidence='Only new native measurements. No historical latency pooling.',
        valid_outer_contexts=valid,
        outside_graph_removal_upper={p:baseline[p]['outside_graph_optimistic_fraction'] for p in valid},
        cost_model_status='PARTIAL_CALIBRATION_ONLY. No validated lower-TP/PP response curve, so candidate B1 numbers remain unresolved.',
        selected_control='Exact compact out-of-place KV scatter, unchanged TP4',
        reason='Actual40x storage clone at each layer is directly source-correlated. Stage C tests its removability without PP/DP confounds.',
        before_runtime_control=True,
        other_architecture_probes='NOT_RUN. Do not implement an engine or assume dependent refinements fill a pipeline.',
        new_architecture_novelty='NOT_ESTABLISHED. A generic layout correction can be useful without a novel parallel architecture.'))
    print('Screened physical4-GPU TP/DP/PP choices. Selected one exact same-layout copy control.')


if __name__=='__main__':main()
