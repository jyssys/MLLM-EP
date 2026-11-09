"""Calibrate Native batch behavior before held-out dynamic confirmation."""
from .pilot import *


def main():
    prepare();jobs=[]
    for model in ('ouro','huginn'):
        for arm in ('native','zero'):
            cfg=common_pilot(model,f'cap_{model}_{arm}_calibration128',arm,128,2026110900)
            cfg.update(phase='SELECTION_NATIVE_CAPACITY_AND_BATCH_CALIBRATION_NOT_CONFIRMATION',
                adaptive_decode_reserve=True)
            if arm=='native':cfg.update(native_path=False,feature_native_wire_v5=False,atomic_prompt=True)
            save(cfg);jobs.append(cfg['name'])
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];index=queue.index(active)+1
    write(path,queue[:index]+jobs+queue[index:])


if __name__=='__main__':main()
