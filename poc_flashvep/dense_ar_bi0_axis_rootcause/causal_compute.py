"""Compare request-matched source kernels only after both observer gates pass."""
import json
import numpy as np
from .common import ROOT,write
from .causal_replay import extract


def main():
    pairs=[]
    for mode in ('ONLINE','SERIAL_ADMISSION'):
        sides={};inputs={};host={};names={}
        for axis in ('Layer','Token'):
            name=f'RuntimeAPIProfile_{axis}_{mode}_profile_s2026112082_N200_v1';names[axis]=name
            path=ROOT/'analysis'/(name+'_observer_gate.json')
            if not path.exists():continue
            gate=json.loads(path.read_text())
            if not gate['qualified']:continue
            sides[axis]=gate['kernel_measurement'];inputs[axis]=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
            observed=extract(name)
            host[axis]=dict(P_dispatch_to_parent_ready_mean_ms=float(np.mean([r['P_service'] for r in observed['rows']]))*1000,
                D_submit_to_first_client_mean_ms=float(np.mean([r['D_first_service'] for r in observed['rows']]))*1000,
                D_first_to_last_client_mean_ms=float(np.mean([r['D_decode'] for r in observed['rows']]))*1000,
                TTFT_mean_ms=observed['measured_mean_TTFT_ms'],TTFT_p95_ms=observed['measured_p95_TTFT_ms'])
        if len(sides)!=2:continue
        assert inputs['Layer']['arrival_offsets']==inputs['Token']['arrival_offsets']
        assert inputs['Layer']['replay_output_lengths']==inputs['Token']['replay_output_lengths']
        assert inputs['Layer']['dataset_SHA256']==inputs['Token']['dataset_SHA256']
        assert [r['prompt_id'] for r in sides['Layer']['rows']]==[r['prompt_id'] for r in sides['Token']['rows']]
        categories=sorted(set(sides['Layer']['P_named_category_means'])|set(sides['Token']['P_named_category_means']))
        delta=sides['Layer']['P_kernel_union_mean_ms']-sides['Token']['P_kernel_union_mean_ms']
        host_delta=host['Layer']['P_dispatch_to_parent_ready_mean_ms']-host['Token']['P_dispatch_to_parent_ready_mean_ms']
        pairs.append(dict(mode=mode,cut=4,N=200,names=names,host=host,
            source_kernel_mean_ms={axis:sides[axis]['P_kernel_union_mean_ms'] for axis in sides},
            Layer_minus_Token_kernel_ms=delta,Layer_minus_Token_host_source_stage_ms=host_delta,
            observed_kernel_gap_over_source_stage_gap=delta/host_delta if host_delta else None,
            Layer_minus_Token_named_categories_ms={key:sides['Layer']['P_named_category_means'].get(key,0)-sides['Token']['P_named_category_means'].get(key,0) for key in categories},
            evidence='MEASURED_ONLINE',scope='Fixed-output mechanism control,4723tokens,seed2082,QPS2.35,c4,h4,KV128. '
            'One profile and two clean controls per axis/mode. Kernel-category differences are point estimates, not independent-start confidence intervals. '
            'The arithmetic share of an observed stage gap is not a causal percentage from a pure GPU intervention. '
            'No old cut8 serial benefit or old Natural-EOS cut4 queue difference enters this new same-cell subtraction.'))
    write(ROOT/'analysis/qualified_matched_source_compute_comparison.json',dict(results=pairs,
        scope='Only mutually qualified profiles are paired. Unidentified kernel names remain in Other. The identified GEMM category is not complete GEMM time.'))
    print(json.dumps(pairs,indent=2))


if __name__=='__main__':main()
