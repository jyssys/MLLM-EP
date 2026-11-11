"""Short activity-only capacity diagnosis. No new scheduling policy."""
import ctypes,os,time
from .common import *


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    initialize,step,close=StageEngine.__init__,StageEngine._step,StageEngine.close
    marker=ROOT/'raw'/cfg['name']/'measurement_begin.json'

    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs)
        role=engine.stage_config.role
        build=json.loads((ROOT/'tools'/cfg.get('idle_activity_build','activity_build_v2.json')).read_text())
        assert digest(build['library'])==build['library_SHA256']
        handle=ctypes.CDLL(build['library'])
        handle.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        handle.activity_end.argtypes=[ctypes.c_char_p]
        handle.activity_dropped.restype=ctypes.c_uint64
        handle.activity_invalid.restype=ctypes.c_uint64
        handle.activity_count.restype=ctypes.c_size_t
        begin=time.monotonic();code=handle.activity_prepare();assert code==0,code
        if cfg.get('idle_no_midrun_disable'):
            assert handle.activity_large_reserve()==0
        engine._output_idle=dict(role=role,handle=handle,build=build,
            warm_ms=1000*(time.monotonic()-begin),target=None,start=None,stop=None,
            checked=0.,calls=[],marker_poll_ms=50.)
        return result

    def stamp(state):
        value=ctypes.c_uint64();a=time.monotonic()
        code=state['handle'].activity_timestamp(ctypes.byref(value));b=time.monotonic()
        assert code==0,code
        return dict(cupti_ns=value.value,host_lower=a,host_upper=b,host_midpoint=(a+b)/2)

    def stepped(engine):
        state=getattr(engine,'_output_idle',None)
        if state is not None:
            now=time.monotonic()
            if state['target'] is None and now-state['checked']>=.05:
                state['checked']=now
                if marker.exists():
                    begin=json.loads(marker.read_text())['time']
                    state['target']=begin+cfg['idle_capture_offset_s']
            if state['target'] is not None and state['start'] is None and now>=state['target']:
                a=time.monotonic();code=state['handle'].activity_resume();b=time.monotonic()
                assert code==0,code
                state['calls'].append(dict(operation='enable',duration_ms=1000*(b-a)))
                state['start']=stamp(state)
            if state['start'] is not None and state['stop'] is None and now>=state['target']+cfg['idle_capture_seconds']:
                state['stop']=stamp(state)
                # Disable delivery without flushing or exporting during Serving.
                if not cfg.get('idle_no_midrun_disable'):
                    a=time.monotonic();code=state['handle'].cuptiActivityDisable(10);b=time.monotonic()
                    assert code==0,code
                    state['calls'].append(dict(operation='disable',duration_ms=1000*(b-a)))
        return step(engine)

    def closed(engine):
        state=getattr(engine,'_output_idle',None)
        if state is not None and state['start'] is not None:
            engine.model_runner.synchronize()
            if state['stop'] is None:state['stop']=stamp(state)
            path=ROOT/'raw'/cfg['name']/f'idle_kernels_{state["role"]}.csv'
            begin=time.monotonic();code=state['handle'].activity_end(str(path).encode())
            write(path.with_suffix('.json'),dict(role=state['role'],start=state['start'],stop=state['stop'],
                target=state['target'],status=code,calls=state['calls'],warm_ms=state['warm_ms'],
                dropped=state['handle'].activity_dropped(),invalid=state['handle'].activity_invalid(),
                records=state['handle'].activity_count(),build=state['build'],
                export_after_outputs_ms=1000*(time.monotonic()-begin),
                no_midrun_disable=bool(cfg.get('idle_no_midrun_disable')),
                scope='Short actual concurrent-kernel capture. All streams use interval union. No memcpy or hardware resource counters. No CUDA Event observer. Not a whole-cohort idle measurement.'))
        return close(engine)
    StageEngine.__init__,StageEngine._step,StageEngine.close=initialized,stepped,closed


def prepare():
    initialize()
    primary=REPO/'poc_flashvep/reports/huginn_loop28_decode_tail_diagnosis_final_summary.md'
    assert primary.exists(),'Finish the primary diagnosis before additional GPU work'
    write(ROOT/'state/primary_diagnosis_complete.json',dict(time=time.time(),report=str(primary),sha256=digest(primary)))
    names=[]
    for cap in (16,4):
        source=f'pipeline7_output{cap}_zero_seed2026110940'
        original=json.loads((OLD_ROOT/'configs'/(source+'.json')).read_text())
        parent=OLD_ROOT/'raw'/original['required_qualification']/'qualification_with_frozen_reference.json'
        qualification=json.loads(parent.read_text());assert qualification['passed']
        for mode in ('clean','repeat','capture'):
            cfg=dict(original);name=f'idle_output{cap}_{mode}_n64'
            cfg.update(name=name,count=64,request_caps=original['request_caps'][:64],
                prompt_ids=original['prompt_ids'][:64],arrival_offsets=original['arrival_offsets'][:64],
                phase='OUTPUT_CAP_ONLY_SHORT_IDLE_DIAGNOSIS',required_qualification=None,
                diagnosis_parent=dict(configuration=source,qualification=str(parent),qualification_sha256=digest(parent)),
                diagnosis_events=False,output_idle_capture=mode=='capture',idle_capture_offset_s=15.,
                idle_capture_seconds=5.,timeout=420,startup_timeout=240,clean_confirmation=False)
            save(cfg);names.append(name)
    order=['idle_output16_clean_n64','idle_output4_clean_n64',
        'idle_output4_capture_n64','idle_output16_capture_n64',
        'idle_output16_repeat_n64','idle_output4_repeat_n64']
    write(ROOT/'state/output_idle_plan.json',dict(jobs=order,
        scope='Complete the Loop28 report first. Output cap is the only workload change. Same pair, pool, source, arrival seed, and first 64 prompts. Short five-second kernel capture is not whole-cohort idle.'))
    print(' '.join(order))


if __name__=='__main__':prepare()
