"""Retry only startup failures in new immutable namespaces."""
from .common import *


def main():
    initialize();names=['n8_huginn_async_native_state8',
        'n8_ouro_async_wire_static_state8','n8_ouro_async_wire_dynamic_state8',
        'n8_huginn_async_wire_static_state8']
    jobs=[]
    for name in names:
        receipt=json.loads((ROOT/'raw'/name/'dispatcher_result.json').read_text())
        assert not receipt['complete']
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());cfg['name']=name+'_v2'
        cfg['retry_reason']='Attach after Native connector factory creation. StageEngine initially has no connector. Original failed snapshots and logs remain unchanged.'
        save(cfg);jobs.append(cfg['name'])
    p=ROOT/'state/queue.json';queue=json.loads(p.read_text())
    active=json.loads((ROOT/'state/active.json').read_text())['name'];i=queue.index(active)+1
    write(p,queue[:i]+jobs+queue[i:])


if __name__=='__main__':main()
