"""List completed, failed, and unexecuted configurations without filling gaps."""
import json
from .common import ROOT
from .analyze import csv_write


def main():
    rows=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or not {'name','model','gpus','cut'}<=set(cfg):continue
        status=ROOT/'raw'/cfg['name']/'status.json'
        state=json.loads(status.read_text()) if status.exists() else dict(status='NOT_RUN')
        for cell in cfg.get('cohorts',[cfg]):
            rows.append(dict(name=cfg['name'],cohort=cell.get('cohort_id','single'),model=cfg['model'],
                             cut=cell['cut'],gpus=','.join(map(str,cfg['gpus'])),GPU_budget=len(cfg['gpus']),
                             requests=cfg['count'],prompt_manifest=cfg['dataset'],chunk_tokens=cfg['chunk'],
                             output_caps=cell.get('request_caps',[cfg['cap']]),state_capacity_tokens=cfg['state_capacity_tokens'],
                             Native_Mixed=False,status=state['status'],error=state.get('error',''),
                             heavy_probe=bool(cfg.get('verify_transfer') or cfg.get('state_probe')),
                             state_trigger=bool(cell.get('ready_background_requests')),
                             config=str(path.relative_to(ROOT))))
    csv_write('experiment_matrix.csv',rows)
    main=[]
    for model in ('ouro','huginn'):
        for budget in (2,4,8):
            for method in ('DP','DP_Chunked','PD_G1','Loop_Static','Loop_Dynamic','Token_Static','Token_Dynamic','Layer_Static','Layer_Dynamic'):
                for workload in ('W1_LG','W1_Natural','W2','W3','W4'):
                    main.append(dict(model=model,GPU_budget=budget,method=method,workload=workload,
                                     status='NOT_RUN_STAGE_GATE',requests='NOT_MEASURED',SLO_thresholds='NOT_CALIBRATED_FOR_MAIN_W1',
                                     stable_SLO_goodput_requests_s='NOT_MEASURED',maximum_stable_offered_rate='NOT_MEASURED',
                                     TTFT_p50_p95_p99='NOT_MEASURED',TPOT_p50_p95_p99='NOT_MEASURED',
                                     E2E_p50_p95_p99='NOT_MEASURED',accuracy='NOT_MEASURED',
                                     diagnostic_scope='Separate finite mechanism/control results exist. They do not fill this main serving table.'))
    csv_write('main_nine_arm_status_matrix.csv',main)
    print('CONFIGURATION_CELLS',len(rows))


if __name__=='__main__':main()
