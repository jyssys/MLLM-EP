"""Four disjoint quality workers, not a live DP placement benchmark."""
import concurrent.futures
import os
import subprocess
from poc_decode_first_depth_aware_dp.contract import *


def main():
    assert read(ROOT/'outputs/reference_smoke_v1/guard_status.json')['success']
    comparison=read(ROOT/'outputs/reference_smoke_v1/reference_comparison.json')
    limits=comparison['limits']
    assert all(comparison[k]<=limits[k] for k in ['logits_max_abs','logits_mean_abs','KV_relative_L2'])
    # A numerical pass with a natural-prefix mismatch permits only this Stage-0
    # quality diagnostic. It does not qualify a live DP/oracle performance claim.
    write(ROOT/'artifacts/quality_pilot_preflight.json',dict(stage='STAGE_0_ONLY',
        numerical_limits_pass=True,natural_prefix_equal=comparison['natural_ids_equal'],
        reference_status='PASS' if comparison['natural_ids_equal'] else 'REFERENCE_PARTIAL',
        scheduling_or_oracle_claim_allowed=False,quality_thresholds_unchanged=True))
    assert read(ROOT/'outputs/runtime_smoke_v6/status.json')['actual_online_skipping']
    manifest=ROOT/'artifacts/quality_pilot_controller.json'
    assert not manifest.exists()
    def run(gpu,policy,part):
        command=[REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp.scripts.gpu_session','--gpus',str(gpu),
            '--output',f'outputs/quality_{policy}_part{part}_v1','--timeout','7200','--',PYTHON,'-m',
            'poc_decode_first_depth_aware_dp.scripts.quality_run','--policy',policy,'--part',str(part)]
        result=subprocess.run(command,cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK))
        return dict(gpu=gpu,policy=policy,part=part,command=command,return_code=result.returncode)
    assignments=[(4,'fixed',0),(5,'fixed',1),(6,'dynamic',0),(7,'dynamic',1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda args:run(*args),assignments))
    write(manifest,dict(assignments=results,all_success=all(r['return_code']==0 for r in results),
        scope='Independent Stage-0 quality shards. Not a measured DP4 scheduling or load-balancing result.'))
    assert all(r['return_code']==0 for r in results)


if __name__=='__main__':main()
