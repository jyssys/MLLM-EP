"""Retain native grouped progress before a request's own negotiation target."""
import copy
from .common import *


def main():
    initialize();active=json.loads((ROOT/'state/active.json').read_text())['name']
    for count in (8,32):
        old=f'r6_huginn_actual_mixed_state{count}'
        cfg=copy.deepcopy(json.loads((ROOT/'configs'/(old+'.json')).read_text()))
        cfg.update(name=f'r7_huginn_actual_mixed_state{count}',phase='MIXED_DEPTH_SOURCE_GROUP_BOUNDARY_RECOVERY_FULL_STATE',
            source_boundary_scope='A grouped batch can complete an intermediate depth below this requests target. Continue through the Native callback. Offer only at the recorded target. A skipped target remains an error.')
        save(cfg)
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text());index=queue.index(active)+1
    write(path,queue[:index]+['r7_huginn_actual_mixed_state8']+queue[index:])
    print('r7_huginn_actual_mixed_state8')


if __name__=='__main__':main()
