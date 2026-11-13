"""Freeze scalar work budgets from discovery before fresh confirmation."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics,records


def main():
    initialize();name='pilot_Revised_O16_N64_q4p8_s21_v1'
    rows=records(ROOT/'raw'/name/'requests.jsonl');assert len(rows)==64
    positive=[r for r in rows if r['completed_D_prefill_layers']]
    fraction=len(positive)/64;work=sum(r['completed_D_prefill_layers'] for r in rows)/64
    freeze=ROOT/'configs/discovery_matched_budgets.json'
    value=dict(source=name,source_request_file_SHA256=digest(ROOT/'raw'/name/'requests.jsonl'),
        request_fraction=fraction,layers_per_request=work,requests=len(rows),
        actual_migrated_requests=len(positive),actual_migrated_layers=int(work*64),
        scope='Freeze viewed discovery scalar budgets. Fresh outcomes do not choose these budgets. Equal caps do not imply equal realized work.')
    if freeze.exists():assert json.loads(freeze.read_text())==value
    else:write(freeze,value)
    jobs=[]
    for label,arm,changes in (
        ('Static8_ID_fraction','STATIC',dict(static_layers=8,migration_fraction=fraction,static_fraction_axis='REQUEST_ID_BLOCKS',static_request_block_size=4)),
        ('Revised_same_caps','Revised',dict(request_budget_fraction=fraction,layer_budget_per_request=work)),
        ('Static12_ID50','STATIC',dict(static_layers=12,migration_fraction=.5,static_fraction_axis='REQUEST_ID_BLOCKS',static_request_block_size=4)),
        ('Revised_layer6_cap','Revised',dict(layer_budget_per_request=6.)),
        ('Revised_request50_cap','Revised',dict(request_budget_fraction=.5))):
        c=base(f'matched_{label}_O16_N64_q4p8_v1',arm=arm,cap=16,phase='FROZEN_DISCOVERY_MATCHED_BUDGET_CONTROL',**changes)
        save(c);jobs.append(c)
    results=[]
    try:
        for cfg in jobs:
            if not execute(cfg['name'],restore=False):break
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',cfg['name'],True)
            actual=records(ROOT/'raw'/cfg['name']/'requests.jsonl')
            row=dict(name=cfg['name'],output=output,metrics=metrics(cfg['name']),
                actual_migrated_requests=sum(r['completed_D_prefill_layers']>0 for r in actual),
                actual_migrated_layers=sum(r['completed_D_prefill_layers'] for r in actual))
            results.append(row);write(ROOT/'state/matched_control_progress.json',dict(results=results))
            print('MATCHED',row['name'],output['passed'],row['actual_migrated_requests'],row['actual_migrated_layers'],flush=True)
            if not output['passed']:break
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
