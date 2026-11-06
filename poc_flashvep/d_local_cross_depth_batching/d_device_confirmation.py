"""Confirm the strongest micro control on the physical PD decode device."""
import time
from .prepare import *
from .run import execute


def main():
    initialize()
    write(ROOT/'state/micro_device_mapping.json',dict(time=time.time(),
        initial_micro='Visible device 0 maps to physical GPU 0 for Ouro and 4 for Huginn. These are standalone tests, not the serving D devices.',
        confirmation='Reverse the visible pair to use physical GPU 1 for Ouro and 5 for Huginn. No concurrent PD job runs.',
        scope='A same-SKU standalone micro result does not qualify online PD slack. Keep the mapping explicit.'))
    for model in ('ouro','huginn'):
        gate=ROOT/'raw'/f'micro_{model}_matched_c248_v1/micro_summary.json'
        while not gate.exists():
            receipt=gate.parent/'dispatcher_result.json'
            if receipt.exists() and not json.loads(receipt.read_text()).get('complete'):break
            if time.time()>=LONG_START_CUTOFF:return
            time.sleep(10)
        if not gate.exists() or not json.loads(gate.read_text())['all_micro_states_exact']:continue
        cfg=configuration(model)
        cfg.update(name=f'opportunity_{model}_c248_d_device_v1',worker='operator_micro',micro_opportunity=True,
            gpus=list(reversed(cfg['gpus'])),opportunity_chunk=248,opportunity_token_budget=512,
            opportunity_graph_rows=512,opportunity_strict_join=True,opportunity_same_align=True)
        save(cfg);execute(cfg['name'])


if __name__=='__main__':main()
