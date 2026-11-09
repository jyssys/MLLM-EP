"""Recheck the zero-suffix native reservation before Token serving tests."""
import copy
from .common import *


def main():
    assert RESUMPTION is not None
    initialize();active=json.loads((ROOT/'state/active.json').read_text())['name'];jobs=[]
    for count in (8,32):
        old=f'r3_ouro_actual_token_state{count}'
        cfg=copy.deepcopy(json.loads((ROOT/'configs'/(old+'.json')).read_text()))
        cfg.update(name=f'r4_ouro_actual_token_state{count}',
            phase='TOKEN_NATIVE_ZERO_SUFFIX_RESERVATION_RECOVERY_FULL_STATE',
            reservation_recovery='Use qualified Native reserve for k=T. Keep every invalid partial-frontier guard.')
        save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text());index=queue.index(active)+1
    write(p,queue[:index]+jobs+[name for name in queue[index:] if name not in jobs])
    print(json.dumps(jobs))


if __name__=='__main__':main()
