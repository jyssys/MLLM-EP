"""Measure prepared D through the actual native PD coordinator and worker."""
import argparse
import statistics
from .common import *
from .evidence import records,rows,status,full_work
from .families import execution_family,effective_asynchronous


def plan():
    cases=[('ouro',8192,8,'qual_ouro_d128_kv25088_l8192_o32_v1'),
        ('huginn',3500,3,'qual_huginn_d128_kv28288_l3500_o32_n8_v1')];new=[]
    for model,length,batch,reference in cases:
        if status(reference)!='COMPLETE':continue
        for start in range(3):
            source=json.loads((ROOT/'configs'/(reference+'.json')).read_text())
            version='v2' if model=='huginn' else 'v1'
            source.update(name=f'native_D_{model}_l{length}_b{batch}_s{start}_{version}',phase='independent_native_D',
                worker='prepared_d_worker',capacity_mode='NATIVE_PREPARED_D',count=batch,P_slots=batch,
                request_caps=[32]*batch,prompt_ids=source['prompt_ids'][:batch],arrival_offsets=[0.]*batch,
                prepare_D_only=True,prepared_D_arrival_offsets=[0.]*batch,D_ready_pattern='ALL_READY_TOGETHER',
                verify_transfer=False,state_probe=False,independent_start=start,gpus=[2*start,2*start+1],
                native_capacity_reference=reference,native_capacity_control=True,
                capacity_scope='Original native NIXL import and first Coda. Preparation excluded. The source must retain this legal cohort until commit. Not an online rate.')
            source['data_scope']=dict(source.get('data_scope',{}),offered_count=batch)
            new.append(save_config(source))
    p=ROOT/'configs/study_plan.json';old=json.loads(p.read_text())['configs'];write(p,dict(configs=list(dict.fromkeys(new+old))))
    write(ROOT/'state/native_D_capacity_plan.json',dict(configs=new));print('NATIVE_D_SERVICE_CONTROLS',len(new))


def analyze():
    from .analyze import save_csv
    from .uncertainty import bootstrap
    samples=[];groups={}
    for p in (ROOT/'configs').glob('native_D_*.json'):
        cfg=json.loads(p.read_text());name=cfg['name'];folder=ROOT/'raw'/name
        if status(name)!='COMPLETE' or not (folder/'prepared_D_window.json').exists():continue
        window=json.loads((folder/'prepared_D_window.json').read_text());values=records(folder/'prepared_D_requests.jsonl')
        reference={r['sample_id']:r for r in rows(cfg['native_capacity_reference'])}
        parity=len(values)==cfg['count'] and all(r['output_ids']==reference[r['sample_id']]['output_ids'] for r in values)
        work=full_work(name);duration=window['end']-window['start']
        constructed={v['role']:v for v in [json.loads(p.read_text()) for p in folder.glob('effective_execution_*.json')]}
        value=dict(name=name,model=cfg['model'],length=cfg['length'],output_cap=cfg['cap'],batch=cfg['count'],
            execution_family=execution_family(cfg),asynchronous=effective_asynchronous(cfg),requested_asynchronous=cfg['asynchronous'],graph_rows=cfg.get('graph_rows',cfg['capacity']),
            execution_flag_scope='SOURCE_AUDIT_HUGINN_SYNC_ONLY' if cfg['asynchronous'] and cfg['model']=='huginn' else 'REQUESTED_AND_SUPPORTED_CONFIGURATION',
            D_token_budget=cfg['D_token_budget'],D_kv_blocks=cfg['D_kv_blocks'],gpus=cfg['gpus'],independent_start=cfg['independent_start'],
            prepared_D_native_service_s=duration,finite_requests_per_s=len(values)/duration,exact_output_parity=parity,
            full_depth_work_pass=work['passed'],passed=parity and work['passed'],
            evidence='MEASURED_NATIVE_PREPARED_D_WITH_FIRST_CODA',scope='Finite all-ready native PD D service. P preparation and transfer precede the timer. Not staggered online capacity.')
        samples.append(value)
        if 'decode' in constructed:
            value.update(asynchronous=constructed['decode']['effective_async_scheduling'],graph_rows=constructed['decode']['effective_graph_max_rows'],
                execution_flag_scope='MEASURED_STAGE_STARTUP_CONFIGURATION')
        if value['passed']:
            key=(cfg['model'],cfg['length'],cfg['cap'],cfg['count'],cfg['D_token_budget'],cfg['D_kv_blocks'],execution_family(cfg));groups.setdefault(key,[]).append(value)
    summary=[]
    for key,values in groups.items():
        rates=[v['finite_requests_per_s'] for v in values];low,high=bootstrap(rates)
        summary.append(dict(model=key[0],length=key[1],output_cap=key[2],batch=key[3],D_token_budget=key[4],D_kv_blocks=key[5],
            execution_family=key[6],asynchronous=values[0]['asynchronous'],graph_rows=values[0]['graph_rows'],
            independent_starts=len(values),finite_requests_per_s_mean=statistics.mean(rates),bootstrap95_low=low,bootstrap95_high=high,
            source_runs=[v['name'] for v in values],evidence='MEASURED_FINITE_NATIVE_D_SERVICE',
            scope='Native coordinator and same small D workspace. Includes first Coda. N3 starts remain a small inferential sample. All-ready is not online capacity.'))
    save_csv('native_D_capacity_samples.csv',samples);save_csv('native_D_capacity_summary.csv',summary)
    write(ROOT/'native_D_capacity_summary.json',summary);return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','analyze'));args=parser.parse_args()
    if args.action=='plan':plan()
    else:analyze()


if __name__=='__main__':main()
