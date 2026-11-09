"""Do not repeat the measured unsafe multi-shape HBM envelope."""
from .predictive32 import *
import signal


def main():
    prepare();path=ROOT/'state/queue.json';queue=json.loads(path.read_text());skipped=[]
    for name in queue:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        if (cfg['model']=='huginn' and cfg.get('intervention_warmup_widths')
                and cfg.get('D_kv_blocks')==26752 and not (ROOT/'raw'/name/'dispatcher_result.json').exists()):
            skipped.append(name)
    active_record=json.loads((ROOT/'state/active.json').read_text());active=active_record.get('name')
    index=queue.index(active)+1 if active in queue else 0
    insertion=sum(name not in skipped for name in queue[:index])
    if active in skipped:
        pid=active_record['pid'];proc=Path('/proc',str(pid));config_path=ROOT/'configs'/(active+'.json')
        if proc.exists():
            assert proc.stat().st_uid==os.getuid() and os.getsid(pid)==pid
            assert str(config_path) in (proc/'cmdline').read_bytes().decode().split('\0')
            write(ROOT/'raw'/active/'operator_stop.json',dict(time=time.time(),session=pid,
                reason='KNOWN_UNSAFE_MULTI_SHAPE_HBM_ENVELOPE',scope='Stop only this registered scientific session. Preserve partial data. Reserve workspace in a new namespace.'))
            os.killpg(pid,signal.SIGTERM)
    write(ROOT/'state/unsafe_memory_deferred.json',dict(names=skipped,
        reason='Measured D OOM in Huginn multi-shape warm-up. Original KV pool used 67.35 GiB before graph and continuation buffers. Original configurations remain immutable. New matched controls use 23552 D blocks.'))
    queue=[name for name in queue if name not in skipped];write(path,queue);jobs=[]
    for arm in ('native','zero'):
        cfg=configured('huginn',f'ws_huginn_{arm}_clean32',arm,32)
        save(cfg);jobs.append(cfg['name'])
    for safety in ('relative','absolute'):
        for arm in ('loop','tile'):
            cfg=prototype('huginn',arm,safety)
            cfg['name']+='_'+ 'workspace_v2'
            save(cfg);jobs.append(cfg['name'])
    write(path,queue[:insertion]+jobs+queue[insertion:])


if __name__=='__main__':main()
