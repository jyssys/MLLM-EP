"""Prepare separate CUDA Event diagnostics after marker-free recovery qualification."""
import copy
from .common import *


def main():
    initialize();jobs=[]
    for model in ('ouro','huginn'):
        selection=ROOT/'state/earlier_frontier_selection_result.json'
        expanded=model=='huginn' and selection.exists() and json.loads(selection.read_text())['expanded_selected']
        source='earlier_huginn_loop_selection32' if expanded else f'secondary_{model}_loop_selection32'
        prefix='recovery_eventsx' if expanded else 'recovery_events'
        cfg=copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
        cfg.update(name=f'{prefix}_{model}_loop32',sparse_device_trace=True,sparse_decode_period=1,
            observe_device_execution=False,observe_service_timing=False,
            phase='SEPARATE_CUDA_EVENT_ENCLOSURE_DIAGNOSTIC_NOT_MARKER_FREE_PERFORMANCE')
        save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/recovery_event_jobs.json',jobs);print(' '.join(jobs))


if __name__=='__main__':
    main()
