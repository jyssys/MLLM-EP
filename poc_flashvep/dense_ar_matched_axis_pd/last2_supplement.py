"""Add the requested Last2 pair without changing the original matrices."""
import argparse
import collections
import json
import time
from .common import ROOT, initialize, save, write
from .campaign import key, run_job
from .configuration import base, matched_token_plan
from .confirmation import SEEDS
from .prepare import qualification
from .qualification import gate
from .run import execute


def performance(method, cap, qps, seed, stage):
    assert method in ('Layer', 'Token')
    name='last2_'+key(method, 2, 144, cap, qps, seed, stage)
    # The frozen constructor contains only the original Token4/8/12 defaults.
    # Override the declarative partition before saving or executing anything.
    cfg=base(name, method, 4 if method=='Token' else 2, 144, cap, qps, seed,
        'matched144.json', phase='USER_ADDED_LAST2_'+stage.upper(),
        D_admission_batch=8, reference_name='HISTORICAL_original_outputs')
    cfg.update(static_layers=2, layer_cut=34, matrix_stage=stage,
        common_prospective_KV_reservation=True, execution_contract_version=3,
        last2_supplement=True)
    if method=='Token':
        cfg.update(token_suffix=448,
            token_suffix_by_request=matched_token_plan(2, 144, group_size=1))
    return cfg


def state_configuration(method, count, cap=16):
    cfg=qualification(method, count, 4 if method=='Token' else 2 if method=='Layer' else 0,
        version=112)
    cfg.update(name=f'last2_qual_{method}_Last{2 if method!="Native" else 0}_N{count}_O{cap}_v3',
        cap=cap, request_caps=[cap]*count,
        static_layers=2 if method!='Native' else 0, layer_cut=34 if method!='Native' else 36,
        D_admission_batch=8, witness_hidden_partition_tokens=[432,448],
        last2_supplement=True)
    if method=='Token':
        cfg.update(token_suffix=448, token_suffix_by_request=[(432,448)[i%2] for i in range(count)])
    return cfg


def matrix():
    configs=[]
    for qps in (2.,6.,10.,4.,8.):
        for cap in (200,16,64,32,128):
            axes=('Layer','Token') if int(qps+cap)%4 else ('Token','Layer')
            configs.extend(performance(axis,cap,qps,2026111780,'discovery') for axis in axes)
    for qps in (3.,5.,7.,9.):
        configs.extend(performance(axis,200,qps,2026111780,'motivation_odd') for axis in ('Layer','Token'))
    for rep,seed in enumerate(SEEDS):
        for index,cap in enumerate((200,16,64,32,128)):
            axes=('Layer','Token') if (rep+index)%2==0 else ('Token','Layer')
            configs.extend(performance(axis,cap,6.,seed,'confirmation') for axis in axes)
    assert len(configs)==88
    return configs


def freeze():
    initialize()
    configs=matrix()
    value=dict(count=144,B=8192,S=1024,D_admission_cap=8,layers=2,
        token_suffix_histogram=dict(collections.Counter(matched_token_plan(2,144,group_size=1))),
        outputs=[16,32,64,128,200],coarse_QPS=[2,6,10,4,8],
        Output200_additional_QPS=[3,5,7,9],discovery_seed=2026111780,
        confirmation_QPS=6,confirmation_seeds=list(SEEDS),
        discovery_starts=58,confirmation_starts=30,performance_starts=88,
        qualification_counts=[8,32,128],Output200_qualification_count=8,
        original_matrix_unchanged=True,existing_Last4_Last8_not_repeated=True,
        inference_contract='Existing frozen revision3. No inference adapter change.',
        scope='The user requested this supplemental pair after the Last2/Last4 serial diagnostic. This is not part of the original design freeze. Reuse the original prompt pool, traces, SLO, GPU pair and common admission policy. Keep discovery and confirmation separate. Reuse existing Native and Agg starts. Do not claim unseen prompts.')
    path=ROOT/'state/last2_supplement_design.json'
    if path.exists():
        assert {k:v for k,v in json.loads(path.read_text()).items() if k!='frozen_epoch'}==json.loads(json.dumps(value))
    else:
        write(path,dict(frozen_epoch=time.time(),**value))
    write(ROOT/'state/last2_supplement_manifest.json',dict(expected=[c['name'] for c in configs],
        discovery=[c['name'] for c in configs if c['matrix_stage']!='confirmation'],
        confirmation=[c['name'] for c in configs if c['matrix_stage']=='confirmation'],total=len(configs)))
    for cfg in configs:
        save(cfg)
    from ..dense_ar_layer_token_pd.structural import counts
    plan=matched_token_plan(2,144,group_size=1)
    per=[counts(8000,2,s) for s in plan]
    total={field:sum(r[field] for r in per) for field in per[0]}
    all_pairs=36*8000*8001//2*144
    total.update(layer_P_attention_pairs=all_pairs-total['layer_D_attention_pairs'],
        token_P_attention_pairs=all_pairs-total['token_D_attention_pairs'],
        total_attention_pairs=all_pairs,layer_P_GEMM_rows=144*8000*34,
        token_P_GEMM_rows=36*sum(8000-s for s in plan))
    assert total['layer_D_GEMM_rows']==total['token_D_GEMM_rows']==2304000
    assert total['layer_P_GEMM_rows']==total['token_P_GEMM_rows']
    write(ROOT/'analysis/last2_structural_work.json',dict(rows=[dict(layers=2,count=144,
        token_suffix_histogram=dict(collections.Counter(plan)),totals=total,
        D_token_to_layer_attention_pair_ratio=total['token_D_attention_pairs']/total['layer_D_attention_pairs'])],
        evidence='STRUCTURAL_ACCOUNTING',
        scope='Exact causal pairs and token-layer rows. These are not measured GPU durations. The original isolated operator table has no Last2 shape samples.'))
    return configs


def qualify():
    for count,cap in [(n,16) for n in (8,32,128)]+[(8,200)]:
        reference=state_configuration('Native',count,cap)
        values=[]
        for method in ('Native','Layer','Token'):
            cfg=state_configuration(method,count,cap)
            assert execute(save(cfg)),cfg['name']
            values.append(gate(cfg['name'],reference['name']))
        write(ROOT/'analysis'/f'last2_qualification_N{count}_O{cap}.json',dict(
            time=time.time(),passed=True,names=[r['name'] for r in values],
            evidence='CORRECTNESS_ONLY',common_D_admission_cap=8))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    assert not (ROOT/'state/last2_supplement_cancelled.json').exists(),'The user cancelled this extension. Do not resume it.'
    configs=freeze()
    if args.prepare_only:
        print('LAST2_SUPPLEMENT_PREPARED',len(configs),flush=True)
        return
    try:
        qualify()
        for index,cfg in enumerate(configs):
            run_job(cfg)
            write(ROOT/'state/last2_supplement_progress.json',dict(time=time.time(),
                completed=index+1,total=len(configs),last=cfg['name']))
        write(ROOT/'state/last2_supplement_complete.json',dict(time=time.time(),
            total=len(configs),passed=True))
    except BaseException as error:
        write(ROOT/'state/last2_supplement_stopped.json',dict(time=time.time(),error=repr(error),
            scope='Preserve the failed run. Do not silently retry or change a saved configuration.'))
        raise


if __name__=='__main__':
    main()
