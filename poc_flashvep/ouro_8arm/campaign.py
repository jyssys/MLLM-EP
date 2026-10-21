"""Restartable bounded stages. A failed probe prevents the related quality run."""
import argparse
from .contract import *
from .launch import execute
from .lease import restore


def run(label,module,gpus=(4,5,6,7),args=(),native=False,timeout=14400,distributed=False):
    marker=ROOT/'raw'/label/'complete.json'
    completion=ROOT/'commands'/f'{label}_completion.json'
    if marker.exists() and read(marker).get('success') and completion.exists() and read(completion)['code']==0:return
    if set(gpus)=={4}:restore([5,6,7])
    interpreter=VLLM_PYTHON if native else PYTHON
    prefix=[interpreter,'-u']
    if distributed:prefix+=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4']
    argv=prefix+['-m','poc_flashvep.ouro_8arm.'+module,'--label',label]+list(args)
    execute(label,argv,gpus,timeout=timeout,restore_after=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['static-probes','static-quality','static-primary','dynamic-probes','dynamic-quality'],required=True)
    a=p.parse_args()
    try:
        if a.stage=='static-probes':
            if not (ROOT/'numerical/fp32_seams.json').exists():
                execute('FP32_seams_v1',[PYTHON,'-u','-m','poc_flashvep.ouro_8arm.fp32_seams'],[4],restore_after=False,timeout=600)
            if not (ROOT/'numerical/HF_PP_probes.json').exists():
                execute('HF_PP_reference_v1',[PYTHON,'-u','-m','poc_flashvep.ouro_8arm.hf_probe'],[4],restore_after=False,timeout=600)
            for arm in ['S4','L4','O4']:
                run('probe_'+arm+'_v3','request_runtime',ARMS[arm]['GPUs'],['--arm',arm,'--probe','--cap','8','--trace'],timeout=600)
            from .compare_probes import compare
            compare('probe_S4_v3','probe_L4_v3');compare('probe_S4_v3','probe_O4_v3')
            run('probe_T4_graph_v3','tp_runtime',args=['--probe','--cap','8'],distributed=True,timeout=300)
        elif a.stage in ['static-quality','static-primary']:
            from .compare_probes import compare
            compare('probe_S4_v3','probe_L4_v3');compare('probe_S4_v3','probe_O4_v3')
            if a.stage=='static-quality':
                run('quality_S4_pilot_v2','request_runtime',[4],['--arm','S4','--resume-from','quality_S4_pilot_v1'],timeout=21600)
            for arm in ['L4','O4']:
                run('quality_'+arm+'_pilot_v1','request_runtime',ARMS[arm]['GPUs'],['--arm',arm],timeout=21600)
            if a.stage=='static-quality':run('quality_T4_custom_pilot_v1','tp_runtime',distributed=True,timeout=21600)
        elif a.stage=='dynamic-probes':
            run('CDB_SD_smoke_v1','cdb_reference',[4],['--limit','4','--cap','16','--validate-KV'],native=True,timeout=900)
            run('CDB_SD_synthetic_v1','cdb_reference',[4],['--probe','--cap','8','--validate-KV'],native=True,timeout=900)
            status={arm:dict(status='PENDING') for arm in ['SD','LD','OD','TD']}
            for forced in [True,False]:
                suffix='forced_v1' if forced else 'live_v1'
                for arm in ['SD','LD','OD']:
                    if status[arm]['status']=='BLOCKED':continue
                    extra=['--forced-depths','2,3,4'] if forced else []
                    try:
                        run('probe_'+arm+'_'+suffix,'request_runtime',ARMS[arm]['GPUs'],['--arm',arm,'--probe','--cap','8','--trace']+extra,timeout=900)
                        if arm!='SD':
                            from .compare_probes import compare
                            compare('probe_SD_'+suffix,'probe_'+arm+'_'+suffix)
                        status[arm]=dict(status='PASS' if not forced else 'FORCED_PASS')
                    except Exception as error:
                        status[arm]=dict(status='BLOCKED',reason=str(error))
                        write(ROOT/'state/dynamic_probe_status.json',status)
                        if arm=='SD':raise
            write(ROOT/'state/dynamic_probe_status.json',status)
            from .reference_comparison import compare_reference
            compare_reference('CDB_SD_synthetic_v1','probe_SD_live_v1')
            run('probe_T4_graph_v3','tp_runtime',args=['--probe','--cap','8'],distributed=True,timeout=300)
            try:
                run('probe_TD_forced_v1','tp_runtime',args=['--probe','--cap','8','--dynamic','--forced-depths','2,3,4'],distributed=True,timeout=1200)
                from .compare_tp_probe import compare as compare_TP
                compare_TP('probe_SD_forced_v1','probe_TD_forced_v1')
                run('probe_TD_graph_v1','tp_runtime',args=['--probe','--cap','8','--dynamic'],distributed=True,timeout=1200)
                comparisons=compare_TP('probe_SD_live_v1','probe_TD_graph_v1')
                status['TD']=dict(status='BOUNDED_PARITY_WITH_NUMERICAL_WARNING' if all(r['token_exact'] and r['depth_exact'] for r in comparisons) else 'EXPLORATORY_GATE_OR_OUTPUT_DRIFT',
                    qualification='Historical full-model numerical qualification remains unresolved. See both assembled-head KV probe files.')
            except Exception as error:status['TD']=dict(status='BLOCKED',reason=str(error))
            write(ROOT/'state/dynamic_probe_status.json',status)
        else:
            run('CDB_SD_pilot_v1','cdb_reference',[4],['--batch','4'],native=True,timeout=21600)
            status=read(ROOT/'state/dynamic_probe_status.json')
            for arm in ['SD','LD','OD']:
                if status[arm]['status']=='BLOCKED':continue
                run('quality_'+arm+'_pilot_v1','request_runtime',ARMS[arm]['GPUs'],['--arm',arm],timeout=21600)
            if status['TD']['status']!='BLOCKED':run('quality_TD_pilot_v1','tp_runtime',args=['--dynamic'],distributed=True,timeout=21600)
    finally:restore([4,5,6,7])


if __name__=='__main__':main()
