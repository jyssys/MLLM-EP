"""Keep planned, failed, correctness and measured jobs distinct."""
import json
import time
from .common import ROOT,write,digest


def main():
    rows=[]
    observers={}
    for file in sorted((ROOT/'analysis').glob('BI0_*observer*.json')):
        value=json.loads(file.read_text())
        if not isinstance(value,dict) or 'observer_gate_passed' not in value:continue
        name=value.get('names',{}).get('profile')
        if name:observers.setdefault(name,[]).append(dict(path=str(file.relative_to(ROOT)),passed=value['observer_gate_passed']))
    lease=json.loads((ROOT/'state/GPU_lease_current.json').read_text())
    for file in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(file.read_text())
        if not isinstance(cfg,dict) or 'entry_module' not in cfg:continue
        name=cfg['name'];folder=ROOT/'raw'/name
        receipt=json.loads((folder/'dispatcher_result.json').read_text()) if (folder/'dispatcher_result.json').exists() else None
        state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        if receipt is not None:status='COMPLETE' if receipt['complete'] else 'FAILED'
        elif lease.get('status')=='RUNNING' and lease.get('name')==name:status='RUNNING'
        else:status='NOT_RUN' if not folder.exists() else 'INCOMPLETE'
        if status=='COMPLETE':
            if cfg.get('diagnostic_controlled_output_replay'):evidence='DIAGNOSTIC_CONTROLLED_OUTPUT_REPLAY'
            elif cfg.get('correctness_only') or cfg.get('state_capture') or cfg.get('axis_state_capture') or cfg.get('numerical_probe'):evidence='CORRECTNESS_ONLY'
            elif cfg.get('kernel_activity'):
                gates=observers.get(name,[])
                evidence='MEASURED_KERNEL_ACTIVITY' if gates and all(r['passed'] for r in gates) else 'DIAGNOSTIC_OBSERVER_GATE_FAILED' if gates else 'DIAGNOSTIC_PENDING_OBSERVER_GATE'
            elif cfg.get('isolated_continuation_cost') or cfg.get('phase')=='ISOLATED_SERIAL_SERVICE_COST':evidence='ISOLATED_SERIAL_SERVICE_MEASUREMENT'
            elif cfg['entry_module'].endswith(('.isolated_axis','.gemm_api_cost')):evidence='ISOLATED_KERNEL_MEASUREMENT_WITH_CELL_GATES'
            elif cfg['entry_module'].endswith('.natural_reference'):evidence='NATURAL_OUTPUT_REFERENCE'
            else:evidence='MEASURED_ONLINE'
        else:evidence=status
        quality=ROOT/'analysis'/f'{name}_quality_gate.json';gate=json.loads(quality.read_text()) if quality.exists() else {}
        runtime={p.stem:json.loads(p.read_text()) for p in folder.glob('actual_runtime*.json')} if folder.exists() else {}
        rows.append(dict(name=name,status=status,evidence=evidence,method=cfg.get('method'),phase=cfg.get('phase'),
            B=cfg['token_budget'],S=cfg['max_num_seqs'],KV_block_tokens=cfg.get('KV_block_tokens',16),
            declared_gpus=cfg['gpus'],dataset=cfg['dataset'],dataset_SHA256=cfg.get('dataset_SHA256'),
            config_SHA256=digest(file),source_archive_SHA256=receipt.get('source_SHA256') if receipt else None,
            count=cfg.get('count',state.get('completed')),QPS=cfg.get('offered_QPS'),arrival_seed=cfg.get('arrival_seed'),
            actual_runtime=runtime,task_quality_point=gate.get('point_passed'),task_quality_CI=gate.get('CI_passed')))
        rows[-1]['observer_gates']=observers.get(name,[])
        rows[-1]['kernel_activity_roles']=cfg.get('kernel_activity_roles',['prefill','decode']) if cfg.get('kernel_activity') else []
    write(ROOT/'analysis/experiment_register.json',dict(time=time.time(),jobs=rows,
        scope='Raw job completion is not full accuracy, decode-safety or performance qualification. Cell-specific gates remain separate.'))


if __name__=='__main__':main()
