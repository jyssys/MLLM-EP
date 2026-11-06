"""Test final-flush context and buffer volume in an isolated quiet lane."""
from .prepare import *
from .run import execute


def main():
    prepare();results=[]
    for flush_device in (0,1):
        name=f'collector_device1_flush{flush_device}_launches10000'
        cfg=configuration('ouro',name,count=8)
        cfg.update(operator=True,idle_worker='collector_selftest',activity_mode=3,
            selftest_device=1,selftest_flush_device=flush_device,selftest_launches=10000,
            phase='COLLECTOR_COVERAGE_DIAGNOSIS_NOT_SERVING',timeout=180)
        save(cfg);results.append(dict(name=name,complete=execute(name)))
    # A paired small serving test changes final-flush context only.
    for close_context in (False,True):
        name=f'collector_ouro_decode_only_closectx{int(close_context)}_smoke24'
        cfg=configuration('ouro',name,count=24,observer='graph')
        cfg.update(idle_activity_device_context=True,idle_observer_roles=['decode'],
            idle_close_device_context=close_context,idle_queue_stride=16,
            phase='COLLECTOR_COVERAGE_DIAGNOSIS_NOT_CLEAN_SERVING')
        save(cfg);results.append(dict(name=name,complete=execute(name)))
    write(ROOT/'state/collector_diagnosis.json',dict(status='COMPLETE',results=results))


if __name__=='__main__':main()
