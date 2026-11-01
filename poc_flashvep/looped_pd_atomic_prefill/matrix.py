"""Preserve planned, complete, failed, and untested configurations explicitly."""
from .common import *
from .evidence import status,rows
from .analyze import save_csv


def main():
    output=[];actual=[];skip=ROOT/'state/skipped_jobs.json';skipped=json.loads(skip.read_text()) if skip.exists() else {}
    for p in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(p.read_text())
        if not isinstance(cfg,dict) or 'name' not in cfg:continue
        state=skipped.get(cfg['name'],{}).get('status',status(cfg['name']))
        row=dict(name=cfg['name'],model=cfg['model'],length=cfg['length'],output_cap=cfg['cap'],chunk=cfg['chunk'],
            arm=cfg['arm'],phase=cfg['phase'],gpus=cfg['gpus'],P_slots=cfg['P_slots'],D_slots=cfg['D_slots'],
            token_budget=cfg['token_budget'],KV_blocks=cfg['kv_blocks'],target_QPS=cfg.get('arrival_rate'),
            count=cfg['count'],recorded_request_rows=len(rows(cfg['name'])),status=state,
            independent_start=cfg.get('independent_start'),configuration=str(p.relative_to(ROOT)))
        actual.append(row)
    for model,cells in GRID.items():
        for length,cap in cells:
            for label,c in [('C256',256),('C1024',1024),('C2048',2048),('C4096',4096),('FULL',length)]:
                chunk=min(c,length);matching=[r for r in actual if r['model']==model and r['length']==length and r['output_cap']==cap and r['chunk']==chunk and r['phase'] in ('qps_pilot','final_repeat','paper_budget_control')]
                observed=sorted({r['status'] for r in matching})
                output.append(dict(model=model,input_tokens=length,output_tokens=cap,requested_chunk=label,effective_chunk=chunk,
                    alias_of_full=c>=length and label!='FULL',configs=len(matching),status=';'.join(observed) if observed else
                        'NOT_RUN_OPTIONAL_CONTEXT_POOL' if length==16384 else 'NOT_RUN_OR_CAPACITY_PRUNED',
                    scope='The entire requested grid remains visible. Empty entries are not measured results.'))
    save_csv('configuration_matrix.csv',actual);save_csv('requested_search_grid.csv',output)
    print('CONFIGURATIONS',len(actual),'REQUESTED_GRID',len(output));return actual,output


if __name__=='__main__':main()
