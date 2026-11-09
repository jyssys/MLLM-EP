"""Replay the exact failed Huginn N32 cohort without bypassing progress checks."""
import copy
from .common import *


def main():
    initialize();active=json.loads((ROOT/'state/active.json').read_text())['name']
    old='r3_huginn_actual_token_state32'
    cfg=copy.deepcopy(json.loads((ROOT/'configs'/(old+'.json')).read_text()))
    cfg.update(name='r4_huginn_actual_token_state32',phase='TOKEN_READY_TAKE_PROGRESS_RECOVERY_FULL_STATE',
        progress_recovery='A stale readiness observation must not hide runnable Decode. Recheck the chosen cost against the live budget. Preserve the Native no-progress guard.')
    save(cfg);path=ROOT/'state/queue.json';queue=json.loads(path.read_text());index=queue.index(active)+1
    write(path,queue[:index]+[cfg['name']]+queue[index:])
    print(cfg['name'])


if __name__=='__main__':main()
