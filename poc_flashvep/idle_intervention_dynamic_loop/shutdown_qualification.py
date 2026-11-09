"""Use the requested diagnostic shutdown budget in the actual native PDConfig."""
import copy
from .common import *


def main():
    initialize();active=json.loads((ROOT/'state/active.json').read_text())['name']
    old='r6_ouro_actual_mixed_state32';cfg=copy.deepcopy(json.loads((ROOT/'configs'/(old+'.json')).read_text()))
    cfg.update(name='r8_ouro_actual_mixed_state32',phase='FULL_BYTE_NATIVE_SHUTDOWN_CONFIG_RECOVERY',
        shutdown_recovery='The original top-level pd_shutdown_timeout did not override native PDConfig. Deferred hashes consumed most of the default 30-second close window. Resolve the requested 300-second diagnostic budget explicitly.')
    cfg['native_transport_limits']=dict(cfg['native_transport_limits'],shutdown_timeout=cfg['pd_shutdown_timeout'])
    save(cfg);path=ROOT/'state/queue.json';queue=json.loads(path.read_text());index=queue.index(active)+1
    write(path,queue[:index]+[cfg['name']]+queue[index:]);print(cfg['name'])


if __name__=='__main__':main()
