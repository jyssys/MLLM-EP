"""Keep exact measured snapshots stable. Hash them after online decisions finish."""
import copy
from .common import *


def main():
    initialize();active=json.loads((ROOT/'state/active.json').read_text())['name']
    old='r3_ouro_actual_mixed_state8';cfg=copy.deepcopy(json.loads((ROOT/'configs'/(old+'.json')).read_text()))
    cfg.update(name='r5_ouro_actual_mixed_state8',state_hash_defer_until_close=True,
        state_pinned_slots=10,phase='EXACT_CPU_HASH_AFTER_SERVING_DIAGNOSTIC_NOT_CLEAN_LATENCY',
        hash_scope='Copy every measured valid byte under a native local lease. Release the GPU read lease only after D2H completion. Keep CPU buffers immutable until SHA256 completes after serving.')
    save(cfg)
    larger=copy.deepcopy(json.loads((ROOT/'configs/r3_ouro_actual_mixed_state32.json').read_text()))
    larger.update(name='r5_ouro_actual_mixed_state32',state_hash_defer_until_close=True,
        state_pinned_slots=34,phase=cfg['phase'],hash_scope=cfg['hash_scope'])
    save(larger)
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text());index=queue.index(active)+1
    write(path,queue[:index]+[cfg['name']]+queue[index:])
    print(json.dumps(dict(admitted=cfg['name'],pending_N8_gate=larger['name'])))


if __name__=='__main__':main()
