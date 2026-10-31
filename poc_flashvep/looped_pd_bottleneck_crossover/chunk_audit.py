"""Qualify chunk tuning before attributing a bottleneck to partitioning."""
import statistics
from .common import *
from .summary_tables import records,save_csv

def main():
    rows=[];chosen={}
    for model in ('ouro','huginn'):
        base_run='phase_a_ouro_batch8_start0' if model=='ouro' else 'phase_a_huginn_grid_start0'
        path=ROOT/'raw'/base_run/'references.jsonl'
        if not path.exists():continue
        baseline=[r for r in records(path) if r['job']['dataset']=='test_L3072.json' and r['job']['cap']==128]
        if not baseline:continue
        ref=max(baseline,key=lambda r:r['job']['batch'])
        expected={int(k.rsplit('_q',1)[1]):v for k,v in ref['outputs'].items()}
        folders=[p for p in (ROOT/'raw').glob(f'phase_a_{model}_chunk*') if (p/'status.json').exists() and
            json.loads((p/'status.json').read_text())['status']=='COMPLETE']
        references=[r for folder in folders for r in records(folder/'references.jsonl')]
        if not references:continue
        valid={256:True}
        for r in references:
            outputs={int(k.rsplit('_q',1)[1]):v for k,v in r['outputs'].items()}
            matched=sum(v['output_ids']==expected[i]['output_ids'] and v['depths']==expected[i]['depths'] for i,v in outputs.items())
            chunk=r['chunk'];passed=matched==len(outputs);valid[chunk]=valid.get(chunk,True) and passed
            rows.append(dict(model=model,chunk=chunk,batch=r['job']['batch'],matched=matched,total=len(outputs),passed=passed,
                reference_run=base_run,evidence='MEASURED_GPU_OUTPUTS_COMPARED_OFFLINE',
                scope='Cross-chunk exact token/depth parity. Not bitwise KV equivalence or original-HF fidelity.'))
        rates={}
        for root in (ROOT/'raw').glob(f'phase_a_{model}*'):
            if 'event_overhead' in root.name:continue
            for r in records(root/'cohorts.jsonl'):
                if r.get('status')!='COMPLETE' or r['job'].get('cell')!='G2' or not valid.get(r['chunk'],False):continue
                if r['KV_blocks']!=(8192 if model=='ouro' else 18432):continue
                rates.setdefault(r['chunk'],[]).append(r['prefill_host_service_s']/r['batch'])
        if rates:
            per_token={c:statistics.median(v) for c,v in rates.items()}
            best=min(per_token,key=per_token.get)
            chosen[model]=dict(chunk=best,per_request_prefill_s_at_3072=per_token[best],all_chunks_per_request_s=per_token,
                valid_cross_chunk_outputs=valid,
                scope='Isolated service tuning only. Verify native streaming at the same held-out arrival trace before any Dynamic A claim.')
    save_csv(ROOT/'cross_chunk_output_parity.csv',rows)
    write(ROOT/'chunk_tuning_audit.json',chosen);print(json.dumps(chosen))

if __name__=='__main__':main()
