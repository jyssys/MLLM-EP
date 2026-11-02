"""Qualify the public collection toggle before long paired captures."""
from .common import *
from .initial_campaign import run
from .cuda_only_profile import analyze


def main():
    clean='recovery_huginn_3500_32_clean4'
    cfg=json.loads((ROOT/'configs'/(clean+'.json')).read_text())
    cfg.update(name='recovery_huginn_3500_native_bounded3_gate4',cuda_only_profile=True,nsys=True,
        P_only_profiler=False,capture_roles=['prefill'],native_capture_seconds=3.,native_capture_delay_s=0.,
        phase='bounded_stage_profiling',matched_clean=clean,profile_finalization_timeout=300,timeout=600)
    cfg['native_transport_limits']=dict(cfg['native_transport_limits'],shutdown_timeout=300)
    target=ROOT/'configs'/(cfg['name']+'.json')
    if target.exists():assert json.loads(target.read_text())==cfg
    else:write(target,cfg)
    passed=run(cfg['name'])
    result=analyze(cfg['name'],clean) if passed else {}
    write(ROOT/'state/bounded_profile_qualification.json',dict(time=time.time(),run=cfg['name'],complete=passed,analysis=result,
        status='PASS' if passed and result.get('eligible_for_matched_clean') else 'DIAGNOSTIC_ONLY'))
    print(json.dumps(result))
    restore_if_idle()


if __name__=='__main__':main()
