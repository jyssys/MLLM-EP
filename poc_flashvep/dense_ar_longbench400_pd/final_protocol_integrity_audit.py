"""Check frozen plans, real GPU identities, and every final worker contract."""
import csv
import io
import json
import time
from functools import cache
from .common import ROOT,write,digest,preserve,MODEL_REVISION


@cache
def dataset_hash(name):
    return digest(ROOT/'data'/name)


def main():
    preserve()
    plan_path=ROOT/'configs/final_continuous_execution_plan_v1.json'
    main_plan=json.loads(plan_path.read_text())
    main_names={job for c in main_plan['cells'] for job in c['jobs'].values()}
    choice_path=ROOT/'configs/final_continuous_static_choices_frozen_v1.json'
    choice=json.loads(choice_path.read_text())
    assert digest(choice_path)==main_plan['static_choice_SHA256']
    inventory=json.loads((ROOT/'state/environment_initial.json').read_text())
    hardware={}
    for row in csv.DictReader(io.StringIO(inventory['GPUs'])):
        row={k.strip():v.strip() for k,v in row.items()}
        hardware[int(row['index'])]=row
    assert set(hardware)==set(range(8)) and all('H100' in r['name'] for r in hardware.values())
    manifests=('final_continuous_execution_plan_v1.json','controlled_continuous_I3_bridge_frozen_v1.json',
        'hybrid_READ_repair_frozen_v1.json','hybrid_READ_repair_layer_control_extension_v1.json',
        'request_registration_deferral_frozen_v1.json','request_registration_q28_extension_frozen_v1.json',
        'request_registration_controlled_bridge_frozen_v1.json','scoped_compile_confirmation_frozen_v1.json',
        'eager_prefill_native_decode_confirmation_frozen_v2.json')
    names={};missing_plans=[];sources_by_plan={}
    for name in manifests:
        path=ROOT/'configs'/name
        if not path.exists():missing_plans.append(name);continue
        p=json.loads(path.read_text())
        sources_by_plan[name]=p['source_capsule_SHA256']
        if name in ('scoped_compile_confirmation_frozen_v1.json','eager_prefill_native_decode_confirmation_frozen_v2.json'):
            current=ROOT/'references/source_capsules'/p['source_capsule']
            expected=p['source_capsule_SHA256'];seen=set()
            while expected!=main_plan['source_capsule_SHA256']:
                assert expected not in seen;seen.add(expected)
                assert digest(current)==expected
                overlay=json.loads(current.with_suffix('.json').read_text())
                assert overlay['SHA256']==expected
                expected=overlay['base_capsule_SHA256']
                current=current.parent/(overlay.get('base_capsule') or main_plan['source_capsule'])
            assert digest(current)==expected
        elif name.startswith('request_registration_'):
            overlay=json.loads((ROOT/'references/source_capsules'/p['source_capsule']).with_suffix('.json').read_text())
            assert overlay['base_capsule_SHA256']==main_plan['source_capsule_SHA256']
        else:assert p['source_capsule_SHA256']==main_plan['source_capsule_SHA256'],name
        if p.get('original_main_plan_SHA256'):
            assert p['original_main_plan_SHA256']==digest(plan_path)
        if p.get('original_bridge_plan_SHA256'):
            assert p['original_bridge_plan_SHA256']==digest(ROOT/'configs/controlled_continuous_I3_bridge_frozen_v1.json')
        for cell in p['cells']:
            if cell.get('workload')=='Heldout200':assert cell['GPU_pair']==[0,1]
            for arm,job in cell['jobs'].items():
                names[job]=dict(plan=name,arm=arm,cell=cell,source=p['source_capsule_SHA256'],
                    frozen_epoch=p.get('time'))
    stop_file=ROOT/'state/scoped_compile_confirmation_complete_v1.json'
    stopped={r['name']:r for r in json.loads(stop_file.read_text())['results']
        if r['status']=='NOT_RUN' and r.get('reason')} if stop_file.exists() else {}
    stop_file=ROOT/'state/eager_compile_confirmation_v2_complete_v1.json'
    if stop_file.exists():
        stopped.update({r['name']:r for r in json.loads(stop_file.read_text())['results']
            if r['status']=='NOT_RUN' and r.get('reason')})
    rows=[];pending=[];failed=[];not_run=[]
    for name,item in sorted(names.items()):
        folder=ROOT/'raw'/name
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists():
            if name in stopped:not_run.append(stopped[name])
            else:pending.append(name)
            continue
        result=json.loads(receipt.read_text())
        if not result['complete']:failed.append(name);continue
        assert result['source_SHA256']==item['source'],name
        cfg=json.loads((folder/'configuration.json').read_text())
        cfg_path=ROOT/'configs'/(name+'.json')
        assert cfg==json.loads(cfg_path.read_text()),name
        command=json.loads((folder/'command.json').read_text())
        assert command['config_SHA256']==digest(cfg_path),name
        if name in main_names:
            assert command['start_epoch']>=choice['selection_epoch'],name
            assert cfg['final_static_choice_SHA256']==main_plan['static_choice_SHA256'],name
        elif item['frozen_epoch'] is not None:
            assert command['start_epoch']>=item['frozen_epoch'],('Start precedes its declared plan freeze',name)
        cell=item['cell']
        assert cfg['gpus']==cell['GPU_pair'] and cfg['seed']==cell['seed'],name
        assert cfg['dataset']==cell['dataset'] and cfg['dataset_SHA256']==dataset_hash(cfg['dataset'])
        assert cfg['token_budget']==8192 and cfg['max_num_seqs']==1024 and cfg['KV_block_tokens']==128
        assert cfg['batch_invariant'] is False and not cfg['controlled_pipeline']
        provenance=json.loads((folder/'input_provenance.json').read_text())
        assert provenance['model_revision']==MODEL_REVISION
        roles=('replica0','replica1') if cfg['method']=='Agg' else ('prefill','decode')
        workers=[]
        for index,role in enumerate(roles):
            runtime=json.loads((folder/('actual_runtime_'+role+'.json')).read_text())
            gpu=cfg['gpus'][index]
            expected=hardware[gpu]['uuid'].removeprefix('GPU-').lower()
            assert runtime['uuid'].removeprefix('GPU-').lower()==expected,(name,role,gpu,runtime['uuid'])
            assert runtime['scheduler_B']==8192 and runtime['scheduler_S']==1024
            assert runtime['KV_block_tokens']==128 and runtime['chunked_prefill'] is True
            assert runtime['model_layers']==36 and runtime['parameter_dtypes']==['torch.bfloat16']
            options=runtime['options']
            assert options['dtype']=='bfloat16' and options['tensor_parallel_size']==1
            assert options['enable_prefix_caching'] is False and options['kv_cache_memory_bytes']==42*1024**3
            assert options['attention_config']['backend']=='FLASH_ATTN'
            assert not options.get('quantization') and 0<runtime['KV_bytes']<=42*1024**3
            workers.append(dict(role=role,physical_GPU=gpu,UUID=runtime['uuid'],
                actual_KV_bytes=runtime['KV_bytes'],actual_running_cap=runtime['runtime_max_num_running_requests'],
                full_model_layers=runtime['model_layers'],compile_mode=runtime['compile_mode'],
                cudagraph_mode=runtime['cudagraph_mode']))
        rows.append(dict(name=name,passed=True,config_SHA256=digest(cfg_path),dataset_SHA256=cfg['dataset_SHA256'],
            source_SHA256=result['source_SHA256'],model_revision=provenance['model_revision'],workers=workers))
    preserve()
    write(ROOT/'analysis/final_protocol_integrity_audit_v1.json',dict(time=time.time(),
        passed=not pending and not missing_plans,verified=len(rows),rows=rows,pending=pending,
        failed_terminal_not_qualified=failed,missing_plans=missing_plans,
        explicit_NOT_RUN=not_run,
        source_capsule_SHA256=main_plan['source_capsule_SHA256'],
        source_capsules_by_plan=sources_by_plan,
        original_main_plan_SHA256=digest(plan_path),static_choice_SHA256=digest(choice_path),
        evidence='RUNTIME_AND_FROZEN_PROTOCOL_AUDIT',
        scope='Check raw/config/command hashes,discovery-freeze chronology,real physical GPU UUIDs and worker configuration. '
            'Every paired final manifest retains its declared source,pair,seed and dataset. '
            'This does not prove that declared36 layers were visited or that KV/hidden bytes or greedy outputs are correct. '
            'Use the independent state/coverage/lifetime/task-quality gates. It also does not establish decode noninferiority. '
            'Failed and pending runs remain separate. No config,raw result,user checkout or GPU process changes.'))
    print('FINAL_PROTOCOL_INTEGRITY',len(rows),'verified',len(pending),'pending',len(failed),'failed',flush=True)


if __name__=='__main__':
    main()
